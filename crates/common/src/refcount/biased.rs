//! Biased reference counting, after PEP 703 and CPython's free-threaded build.
//!
//! Most objects are only ever touched by the thread that created them, so the
//! count is split in two:
//!
//! * `local` belongs to the owning thread (`tid`). The owner changes it with a
//!   plain load and store — no `lock`-prefixed instruction — and no other
//!   thread ever writes it.
//! * `shared` is changed by every other thread with atomic read-modify-writes.
//!   It may go negative: a thread can release a reference the owner counted.
//!
//! The true count is `local + shared`. The two halves are folded together
//! ("merged") when the owner's `local` reaches zero, or when another thread
//! would take `shared` below zero for the first time. The second case cannot
//! touch `local` from the releasing thread, so that thread hands its reference
//! to the owner through a per-thread queue instead (`QUEUED`), and the owner
//! merges at its next eval-breaker check. Once merged, `tid` is [`UNOWNED`] and
//! `shared` alone holds the count.
//!
//! # The `shared` word
//!
//! ```text
//! [ signed count (usize::BITS - 4 bits) ][ LEAKED ][ PUBLISHED ][ 2-bit state ]
//! ```
//!
//! The state is one of [`INIT`], [`MAYBE_WEAKREF`], [`QUEUED`] or [`MERGED`],
//! as in CPython. [`MAYBE_WEAKREF`] keeps the word nonzero so that the owner's
//! zero-local merge goes through a compare-and-swap, which is what makes a
//! conditional incref from another thread ([`RefCount::safe_inc`]) sound for
//! objects reachable through a weak reference, a lock-free cache or the GC
//! lists.
//!
//! # Immortal objects
//!
//! An immortal object has `tid == IMMORTAL_TID`, which no thread ever holds,
//! so the owner test fails and the next compare on the same loaded value
//! returns before any write. See [`RefCount::make_immortal`].

use crate::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::Cell;

const STATE_MASK: usize = 0b11;
/// No thread but the owner has touched the count at zero; `shared` may still
/// carry balanced traffic from other threads.
const INIT: usize = 0;
/// Other threads may try-incref this object (weakref, lock-free cache, GC
/// list). The owner merges through a CAS rather than freeing on its own.
const MAYBE_WEAKREF: usize = 1;
/// A non-owner released a reference while `shared` was zero and handed it to
/// the owner's merge queue, which now holds that reference.
const QUEUED: usize = 2;
/// `local` has been folded into `shared` and `tid` is [`UNOWNED`].
const MERGED: usize = 3;
/// Object was published to a lock-free cache; memory reclamation is
/// deferred through QSBR so concurrent try-incref readers never touch
/// freed memory. Sticky once set.
const PUBLISHED: usize = 1 << 2;
/// The string pool owns this object (it is interned). Implies immortal.
const LEAKED: usize = 1 << 3;
const SHARED_SHIFT: u32 = 4;
const SHARED_ONE: usize = 1 << SHARED_SHIFT;

/// `tid` of an object no thread owns: its count lives in `shared` alone.
const UNOWNED: u32 = 0;
/// `tid` of an immortal object. No thread is ever given this id.
const IMMORTAL_TID: u32 = u32::MAX;
/// A thread that has not created an object yet.
const TID_UNREGISTERED: u32 = u32::MAX - 1;
/// A thread that has left the interpreter; objects it creates are unowned.
const TID_EXITED: u32 = u32::MAX - 2;
const MAX_TID: u32 = u32::MAX - 3;

/// The strong count an immortal object reports.
///
/// On a 64-bit target this is CPython's `_Py_IMMORTAL_REFCNT`, `UINT_MAX`, so
/// `sys.getrefcount(None)` reports the same 4294967295 CPython does — and it
/// is exactly `GC_REACHABLE`, which is what makes the collector treat an
/// immortal candidate as reachable without a special case.
const IMMORTAL_COUNT: usize = if usize::BITS >= 64 {
    u32::MAX as usize
} else {
    (usize::MAX >> (SHARED_SHIFT + 1)) / 2
};

std::thread_local! {
    /// This thread's owner id. Constant-initialized without a destructor, so
    /// reading it is a single thread-pointer-relative load.
    static CURRENT_TID: Cell<u32> = const { Cell::new(TID_UNREGISTERED) };
    /// This thread's "objects queued" flag while it is registered.
    static CURRENT_PENDING: Cell<*const AtomicBool> = const { Cell::new(core::ptr::null()) };
    static EXIT_GUARD: ExitGuard = const { ExitGuard(Cell::new(false)) };
    /// Set while [`merge_queued_objects`] runs on this thread.
    static MERGING: Cell<bool> = const { Cell::new(false) };
}

#[inline(always)]
fn current_tid() -> u32 {
    CURRENT_TID.with(Cell::get)
}

#[inline(always)]
const fn shared_count(shared: usize) -> isize {
    (shared as isize) >> SHARED_SHIFT
}

#[inline(never)]
#[cold]
#[allow(
    clippy::disallowed_methods,
    reason = "refcount overflow must preserve upstream abort semantics"
)]
fn refcount_overflow() -> ! {
    std::process::abort()
}

/// How a thread that queues an object asks its owner to merge soon: the
/// interpreter trips the owner's own eval breaker, so threads with nothing
/// queued keep their fast path.
pub trait OwnerWakeup: Send + Sync {
    /// Called with the registry lock held; must not touch a reference count.
    fn wake(&self);
}

struct Owner {
    /// Objects handed over by other threads, each carrying one reference.
    objects: Vec<usize>,
    /// Whether `objects` is nonempty, readable without the registry lock.
    pending: Arc<AtomicBool>,
    wakeup: Option<Arc<dyn OwnerWakeup>>,
}

struct Registry {
    owners: BTreeMap<u32, Owner>,
    /// Owners whose queue is nonempty.
    pending_owners: usize,
}

static REGISTRY: parking_lot::Mutex<Registry> = parking_lot::Mutex::new(Registry {
    owners: BTreeMap::new(),
    pending_owners: 0,
});
static NEXT_TID: AtomicU32 = AtomicU32::new(1);

/// Deregisters the thread when its thread-locals are torn down, for threads
/// that never called [`exit_current_thread`]. Objects still queued then are
/// merged but not freed, since the interpreter may no longer be usable here.
struct ExitGuard(Cell<bool>);

impl Drop for ExitGuard {
    fn drop(&mut self) {
        if self.0.get() {
            exit_current_thread_impl(None);
        }
    }
}

/// Give the current thread an owner id. Returns [`UNOWNED`] when it cannot
/// have one (it already exited, or its thread-locals are being torn down), in
/// which case its objects start out merged.
#[cold]
#[inline(never)]
fn register_current_thread() -> u32 {
    let Ok(tid) = CURRENT_TID.try_with(Cell::get) else {
        return UNOWNED;
    };
    if tid != TID_UNREGISTERED {
        debug_assert_eq!(tid, TID_EXITED);
        return UNOWNED;
    }
    if EXIT_GUARD.try_with(|g| g.0.set(true)).is_err() {
        CURRENT_TID.set(TID_EXITED);
        return UNOWNED;
    }
    let tid = NEXT_TID.fetch_add(1, Ordering::Relaxed);
    if tid > MAX_TID {
        // Ids are never reused, so a process that has run four billion
        // threads falls back to unowned objects.
        NEXT_TID.store(MAX_TID + 1, Ordering::Relaxed);
        CURRENT_TID.set(TID_EXITED);
        return UNOWNED;
    }
    let pending = Arc::new(AtomicBool::new(false));
    // The registry keeps the flag alive until `exit_current_thread_impl`,
    // which clears this pointer first.
    CURRENT_PENDING.set(Arc::as_ptr(&pending));
    REGISTRY.lock().owners.insert(
        tid,
        Owner {
            objects: Vec::new(),
            pending,
            wakeup: None,
        },
    );
    CURRENT_TID.set(tid);
    tid
}

/// Take `tid`'s queue (empty when it is not registered).
fn take_queue(registry: &mut Registry, tid: u32) -> Vec<usize> {
    let objects = registry
        .owners
        .get_mut(&tid)
        .map(|owner| {
            owner.pending.store(false, Ordering::Relaxed);
            core::mem::take(&mut owner.objects)
        })
        .unwrap_or_default();
    if !objects.is_empty() {
        registry.pending_owners -= 1;
    }
    objects
}

/// Deregister `tid`, returning its entry. The caller drops the entry after
/// releasing the registry lock: its wakeup may own objects.
fn remove_owner(registry: &mut Registry, tid: u32) -> Option<Owner> {
    let owner = registry.owners.remove(&tid)?;
    if !owner.objects.is_empty() {
        registry.pending_owners -= 1;
    }
    Some(owner)
}

/// Set how other threads wake the current thread when they queue an object
/// for it (`None` while it has no interpreter to run). Registers the thread.
pub fn set_current_thread_wakeup(wakeup: Option<Arc<dyn OwnerWakeup>>) {
    let tid = current_owner_id();
    if tid == UNOWNED {
        return;
    }
    let old = {
        let mut registry = REGISTRY.lock();
        let Some(owner) = registry.owners.get_mut(&tid) else {
            return;
        };
        core::mem::replace(&mut owner.wakeup, wakeup)
    };
    drop(old);
}

/// The current thread's owner id, registering it if needed. Zero when it
/// cannot own objects (it has left the interpreter).
#[must_use]
pub fn current_owner_id() -> u32 {
    let tid = current_tid();
    if tid > MAX_TID {
        register_current_thread()
    } else {
        tid
    }
}

/// Whether a thread other than the current one has objects queued to merge.
pub fn others_have_queued_objects() -> bool {
    let registry = REGISTRY.lock();
    registry.pending_owners > usize::from(has_queued_objects())
}

/// Merge `owner`'s queue on its behalf, pushing each object whose count
/// dropped to zero onto `dead` for the caller to deallocate later.
///
/// A thread that is blocked (in I/O, on a lock) never reaches its eval
/// breaker, so without this the references other threads handed back to it
/// would keep objects alive indefinitely. CPython merges every queue during
/// its stop-the-world collection for the same reason.
///
/// # Safety
/// `owner`'s thread must be parked for the duration of the call (stop the
/// world), so that it writes no `local` count of an object it owns.
pub unsafe fn merge_queued_objects_of(owner: u32, dead: &mut Vec<*const RefCount>) {
    if owner == UNOWNED || owner > MAX_TID {
        return;
    }
    let objects = take_queue(&mut REGISTRY.lock(), owner);
    for ptr in objects {
        // SAFETY: the queue's reference kept the object alive.
        let rc = unsafe { &*(ptr as *const RefCount) };
        if rc.explicit_merge(-1) == 0 {
            dead.push(rc);
        }
    }
}

/// Release the registry lock if a thread that did not survive `fork()` held
/// it.
///
/// # Safety
/// Only in the single-threaded child right after `fork()`, before anything
/// else touches the registry.
pub unsafe fn reinit_after_fork() {
    if REGISTRY.is_locked() {
        unsafe { REGISTRY.force_unlock() };
    }
}

/// Forget the threads that did not survive `fork()`, merging what was queued
/// for them; `dealloc` runs for each object whose count dropped to zero.
/// Objects they still own keep their ids and are merged by the next thread
/// that queues them, as for any exited owner.
///
/// # Safety
/// Only in the single-threaded child after `fork()`, after
/// [`reinit_after_fork`].
pub unsafe fn after_fork_child(mut dealloc: impl FnMut(*const RefCount)) {
    let me = current_tid();
    let objects = {
        let mut registry = REGISTRY.lock();
        let dead: Vec<u32> = registry
            .owners
            .keys()
            .copied()
            .filter(|&tid| tid != me)
            .collect();
        dead.into_iter()
            .filter_map(|tid| remove_owner(&mut registry, tid))
            .collect::<Vec<_>>()
    };
    let objects: Vec<usize> = objects
        .into_iter()
        .flat_map(|owner| owner.objects)
        .collect();
    for ptr in objects {
        // SAFETY: the queue's reference kept the object alive.
        let rc = unsafe { &*(ptr as *const RefCount) };
        if rc.explicit_merge(-1) == 0 {
            dealloc(rc);
        }
    }
}

/// Whether other threads queued objects for the current thread to merge.
#[inline]
pub fn has_queued_objects() -> bool {
    let pending = CURRENT_PENDING.with(Cell::get);
    // SAFETY: non-null only while the registry holds the `Arc`.
    !pending.is_null() && unsafe { &*pending }.load(Ordering::Relaxed)
}

/// Merge every object other threads queued for the current thread, calling
/// `dealloc` on each whose count dropped to zero.
///
/// Called by the owner at an eval-breaker check (and before a collection).
///
/// Not reentrant: a finalizer run by `dealloc` reaches the eval breaker
/// again, and merging there could run a second finalizer while the first
/// holds a lock the second one needs. The outer call picks up whatever was
/// queued meanwhile.
pub fn merge_queued_objects(mut dealloc: impl FnMut(*const RefCount)) {
    if !has_queued_objects() || MERGING.with(|m| m.replace(true)) {
        return;
    }
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            MERGING.with(|m| m.set(false));
        }
    }
    let _reset = Reset;
    while has_queued_objects() {
        let objects = take_queue(&mut REGISTRY.lock(), current_tid());
        for ptr in objects {
            // SAFETY: the queue's reference kept the object alive.
            let rc = unsafe { &*(ptr as *const RefCount) };
            // Subtract the reference the queue held.
            if rc.explicit_merge(-1) == 0 {
                dealloc(rc);
            }
        }
    }
}

/// Stop owning objects on this thread: deregister its id, and merge (and
/// `dealloc`) what other threads queued for it. Objects it still owns keep its
/// id; the next thread to take their `shared` count below zero merges them
/// itself, since no thread will write their `local` again. Objects this
/// thread creates afterwards start out merged.
pub fn exit_current_thread(mut dealloc: impl FnMut(*const RefCount)) {
    exit_current_thread_impl(Some(&mut dealloc));
}

fn exit_current_thread_impl(mut dealloc: Option<&mut dyn FnMut(*const RefCount)>) {
    let Ok(tid) = CURRENT_TID.try_with(Cell::get) else {
        return;
    };
    if tid > MAX_TID || tid == UNOWNED {
        return;
    }
    let _ = CURRENT_PENDING.try_with(|p| p.set(core::ptr::null()));
    let objects = {
        let mut registry = REGISTRY.lock();
        // From here on this thread takes the non-owner path for its own
        // objects, and a thread queueing one of them merges it instead.
        let _ = CURRENT_TID.try_with(|t| t.set(TID_EXITED));
        remove_owner(&mut registry, tid)
    };
    let objects = objects.map(|owner| owner.objects).unwrap_or_default();
    let _ = EXIT_GUARD.try_with(|g| g.0.set(false));
    for ptr in objects {
        // SAFETY: the queue's reference kept the object alive.
        let rc = unsafe { &*(ptr as *const RefCount) };
        if rc.explicit_merge(-1) == 0
            && let Some(dealloc) = dealloc.as_deref_mut()
        {
            dealloc(rc);
        }
    }
}

/// Biased reference count. See the module documentation.
///
/// Layout: `tid: u32`, `local: u32`, `shared: usize` — 16 bytes on a 64-bit
/// target, 12 on a 32-bit one.
pub struct RefCount {
    /// Owning thread, [`UNOWNED`] once merged, or [`IMMORTAL_TID`]. Written by
    /// the owner (merge, immortalize) and read by everyone.
    tid: AtomicU32,
    /// The owner's references. Only the owner writes it, with plain
    /// (relaxed) stores; other threads read it only to report a count.
    local: AtomicU32,
    /// Flags, state and the other threads' (signed) references.
    shared: AtomicUsize,
}

impl Default for RefCount {
    fn default() -> Self {
        Self::new()
    }
}

impl RefCount {
    /// Create a count holding one reference, owned by the current thread.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        let mut tid = current_tid();
        if tid > MAX_TID {
            tid = register_current_thread();
        }
        if tid == UNOWNED {
            Self {
                tid: AtomicU32::new(UNOWNED),
                local: AtomicU32::new(0),
                shared: AtomicUsize::new(SHARED_ONE | MERGED),
            }
        } else {
            Self {
                tid: AtomicU32::new(tid),
                local: AtomicU32::new(1),
                shared: AtomicUsize::new(INIT),
            }
        }
    }

    /// Get current strong count.
    ///
    /// Exact on the owning thread and on a merged object; from another thread
    /// it is a snapshot that may be stale, as a relaxed load always was. The
    /// merge folds `local` into `shared` before clearing `local`, and `local`
    /// is read first here, so a racing merge can only make this over-count.
    #[inline]
    pub fn get(&self) -> usize {
        if self.tid.load(Ordering::Relaxed) == IMMORTAL_TID {
            return IMMORTAL_COUNT;
        }
        let local = self.local.load(Ordering::Acquire) as isize;
        let shared = shared_count(self.shared.load(Ordering::Acquire));
        (local + shared).max(0) as usize
    }

    /// Whether this object lives for the whole process.
    #[inline(always)]
    #[must_use]
    pub fn is_immortal(&self) -> bool {
        self.tid.load(Ordering::Relaxed) == IMMORTAL_TID
    }

    /// Increment strong count
    #[inline(always)]
    pub fn inc(&self) {
        let tid = self.tid.load(Ordering::Relaxed);
        if tid == current_tid() {
            let local = self.local.load(Ordering::Relaxed);
            // One comparison stands in for the two cases that are not an
            // ordinary increment: a count of zero wraps above the bound, a
            // count at the ceiling lands on it.
            if local.wrapping_sub(1) < u32::MAX - 1 {
                self.local.store(local + 1, Ordering::Relaxed);
            } else {
                self.inc_local_uncommon(local);
            }
        } else if tid != IMMORTAL_TID {
            self.inc_shared(1);
        }
    }

    /// The owner-side `inc` cases that are not an ordinary increment.
    ///
    /// At zero the object is being deallocated; like the single-threaded
    /// count, take two so the extra reference keeps the next `dec` from
    /// deallocating it a second time. At the ceiling the reference goes to
    /// `shared` instead.
    #[cold]
    #[inline(never)]
    fn inc_local_uncommon(&self, local: u32) {
        if local == 0 {
            self.local.store(2, Ordering::Relaxed);
        } else {
            self.inc_shared(1);
        }
    }

    #[inline(never)]
    fn inc_shared(&self, n: usize) {
        let old = self.shared.fetch_add(n << SHARED_SHIFT, Ordering::Relaxed);
        if shared_count(old) > (isize::MAX >> SHARED_SHIFT) - n as isize {
            refcount_overflow();
        }
    }

    #[inline(always)]
    pub fn inc_by(&self, n: usize) {
        let tid = self.tid.load(Ordering::Relaxed);
        if tid == current_tid() {
            let local = self.local.load(Ordering::Relaxed);
            if let Some(new) = u32::try_from(n).ok().and_then(|n| local.checked_add(n)) {
                self.local.store(new, Ordering::Relaxed);
                return;
            }
            self.inc_shared(n);
        } else if tid != IMMORTAL_TID {
            self.inc_shared(n);
        }
    }

    /// Increment only if the object is alive. Returns true if successful.
    ///
    /// From a thread other than the owner, an unmerged object whose `shared`
    /// count is zero may have references only in `local`, which that thread
    /// cannot see atomically; unless the object is marked
    /// [`MAYBE_WEAKREF`] this conservatively reports it dead, as CPython's
    /// `_Py_TryIncRefShared` does. Weak references, published caches and the
    /// GC lists all set that state before another thread can reach the object.
    #[inline]
    #[must_use]
    pub fn safe_inc(&self) -> bool {
        let tid = self.tid.load(Ordering::Relaxed);
        if tid == IMMORTAL_TID {
            return true;
        }
        if tid == current_tid() {
            let local = self.local.load(Ordering::Relaxed);
            if local == 0 {
                return false;
            }
            if local < u32::MAX {
                self.local.store(local + 1, Ordering::Relaxed);
            } else {
                self.inc_shared(1);
            }
            return true;
        }
        let mut shared = self.shared.load(Ordering::Relaxed);
        loop {
            let state = shared & STATE_MASK;
            if shared_count(shared) == 0 && (state == INIT || state == MERGED) {
                return false;
            }
            match self.shared.compare_exchange_weak(
                shared,
                shared.wrapping_add(SHARED_ONE),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(cur) => shared = cur,
            }
        }
    }

    /// Decrement strong count. Returns true when count drops to 0.
    #[inline(always)]
    #[must_use]
    pub fn dec(&self) -> bool {
        let tid = self.tid.load(Ordering::Relaxed);
        if tid == current_tid() {
            let local = self.local.load(Ordering::Relaxed);
            debug_assert!(local > 0, "owner released a reference it did not hold");
            let local = local.wrapping_sub(1);
            self.local.store(local, Ordering::Relaxed);
            if local != 0 {
                return false;
            }
            // `local` reached zero: the object is dead unless another thread
            // ever needed the shared half (or marked it try-increfable).
            let shared = self.shared.load(Ordering::Acquire);
            if shared == 0 {
                return true;
            }
            return self.merge_zero_local(shared);
        }
        if tid == IMMORTAL_TID {
            return false;
        }
        self.dec_shared()
    }

    /// The owner's `local` reached zero while `shared` is not plain zero:
    /// give up ownership and fold the state to [`MERGED`]. Returns whether the
    /// merged count is zero.
    #[cold]
    #[inline(never)]
    fn merge_zero_local(&self, mut shared: usize) -> bool {
        // Before the CAS: a zero `tid` must imply a merged count to anyone who
        // then reads `shared`.
        self.tid.store(UNOWNED, Ordering::Relaxed);
        loop {
            let new = (shared & !STATE_MASK) | MERGED;
            match self.shared.compare_exchange_weak(
                shared,
                new,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    debug_assert!(shared_count(new) >= 0);
                    return shared_count(new) == 0;
                }
                Err(cur) => shared = cur,
            }
        }
    }

    /// Release a reference from a thread that does not own the object.
    #[inline(never)]
    fn dec_shared(&self) -> bool {
        let mut shared = self.shared.load(Ordering::Relaxed);
        loop {
            let state = shared & STATE_MASK;
            // The first release that would take an unmerged `shared` below
            // zero is handed to the owner instead of subtracted: the queue
            // then holds this reference until the owner merges.
            let queue = shared_count(shared) == 0 && (state == INIT || state == MAYBE_WEAKREF);
            let new = if queue {
                (shared & !STATE_MASK) | QUEUED
            } else {
                shared.wrapping_sub(SHARED_ONE)
            };
            match self.shared.compare_exchange_weak(
                shared,
                new,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    if queue {
                        return self.queue_for_merge();
                    }
                    if new & STATE_MASK == MERGED && shared_count(new) == 0 {
                        core::sync::atomic::fence(Ordering::Acquire);
                        return true;
                    }
                    return false;
                }
                Err(cur) => shared = cur,
            }
        }
    }

    /// Hand the reference just released by a non-owner to the owning thread.
    /// Returns true if it had to be merged here and the count reached zero.
    #[cold]
    #[inline(never)]
    fn queue_for_merge(&self) -> bool {
        let tid = self.tid.load(Ordering::Acquire);
        if tid == IMMORTAL_TID {
            return false;
        }
        debug_assert_ne!(tid, UNOWNED, "only an owned object can be queued");
        {
            let mut registry = REGISTRY.lock();
            let registry = &mut *registry;
            if let Some(owner) = registry.owners.get_mut(&tid) {
                if owner.objects.is_empty() {
                    registry.pending_owners += 1;
                    owner.pending.store(true, Ordering::Relaxed);
                }
                owner.objects.push(self as *const Self as usize);
                // Every time, not only on the first push: the owner may have
                // cleared a request it took for something else.
                if let Some(wakeup) = &owner.wakeup {
                    wakeup.wake();
                }
                return false;
            }
        }
        // The owner has exited, so nothing writes `local` any more and this
        // thread can merge. `QUEUED` keeps any other thread from doing the
        // same. Subtract the reference the queue would have held.
        self.explicit_merge(-1) == 0
    }

    /// Fold `local` (plus `extra`) into `shared`, mark it merged and give up
    /// ownership. Returns the merged count. Only the owner, or any thread
    /// once the owner has exited, may call this.
    fn explicit_merge(&self, extra: isize) -> isize {
        let local = self.local.load(Ordering::Relaxed) as isize;
        let mut shared = self.shared.load(Ordering::Relaxed);
        let count = loop {
            let count = shared_count(shared) + local + extra;
            let new = ((count as usize) << SHARED_SHIFT) | (shared & (PUBLISHED | LEAKED)) | MERGED;
            match self.shared.compare_exchange_weak(
                shared,
                new,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break count,
                Err(cur) => shared = cur,
            }
        };
        self.local.store(0, Ordering::Release);
        if self.tid.load(Ordering::Relaxed) != IMMORTAL_TID {
            self.tid.store(UNOWNED, Ordering::Release);
        }
        debug_assert!(count >= 0);
        count
    }

    /// Let other threads try-incref this object (see [`Self::safe_inc`]).
    /// The caller must hold a strong reference.
    pub fn set_maybe_weakref(&self) {
        let mut shared = self.shared.load(Ordering::Relaxed);
        while shared & STATE_MASK == INIT {
            match self.shared.compare_exchange_weak(
                shared,
                shared | MAYBE_WEAKREF,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(cur) => shared = cur,
            }
        }
    }

    /// [`Self::set_maybe_weakref`] with a plain store.
    ///
    /// # Safety
    /// No other thread may be able to reach the object yet.
    #[inline]
    pub unsafe fn set_maybe_weakref_unshared(&self) {
        let shared = self.shared.load(Ordering::Relaxed);
        if shared & STATE_MASK == INIT {
            self.shared.store(shared | MAYBE_WEAKREF, Ordering::Relaxed);
        }
    }

    /// Mark this object as leaked (interned). It will never be deallocated.
    ///
    /// This also makes the object immortal, and [`RefCount::dec`] depends on
    /// that: leaked and immortal are separate answers — `is_leaked` means "the
    /// string pool owns this copy" and is what pointer-equality key lookups
    /// read — but every leaked object is immortal.
    pub fn leak(&self) {
        debug_assert!(!self.is_leaked());
        self.shared.fetch_or(LEAKED, Ordering::AcqRel);
        self.make_immortal();
    }

    /// Make this object immortal: every later `inc`/`dec` becomes a branch and
    /// the object is never deallocated. `get` reports [`IMMORTAL_COUNT`], so
    /// `strong_count() == 1` fast paths never fire on it and the collector
    /// treats it as a permanent root.
    ///
    /// Must be called by the thread that owns the object (or on an unowned
    /// one): an owner's in-flight `dec` could otherwise still deallocate it.
    /// Idempotent.
    pub fn make_immortal(&self) {
        debug_assert!({
            let tid = self.tid.load(Ordering::Relaxed);
            tid == IMMORTAL_TID || tid == UNOWNED || tid == current_tid()
        });
        self.tid.store(IMMORTAL_TID, Ordering::Release);
    }

    /// Check if this object is leaked (interned).
    pub fn is_leaked(&self) -> bool {
        (self.shared.load(Ordering::Acquire) & LEAKED) != 0
    }

    /// Mark the object as published to a lock-free cache (sticky). Readers of
    /// the cache try-incref from any thread, so this also sets
    /// [`MAYBE_WEAKREF`].
    #[inline]
    pub fn mark_published(&self) {
        let mut shared = self.shared.load(Ordering::Relaxed);
        loop {
            let mut new = shared | PUBLISHED;
            if new & STATE_MASK == INIT {
                new |= MAYBE_WEAKREF;
            }
            if new == shared {
                return;
            }
            match self.shared.compare_exchange_weak(
                shared,
                new,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(cur) => shared = cur,
            }
        }
    }

    #[inline]
    pub fn is_published(&self) -> bool {
        (self.shared.load(Ordering::Acquire) & PUBLISHED) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    fn raw(rc: &RefCount) -> (u32, u32, usize) {
        (
            rc.tid.load(Ordering::Relaxed),
            rc.local.load(Ordering::Relaxed),
            rc.shared.load(Ordering::Relaxed),
        )
    }

    #[test]
    fn a_new_refcount_is_owned_with_one_local_reference() {
        let rc = RefCount::new();
        let (tid, local, shared) = raw(&rc);
        assert_eq!(tid, current_tid());
        assert!(tid != UNOWNED && tid <= MAX_TID);
        assert_eq!(local, 1);
        assert_eq!(shared, INIT);
        assert_eq!(rc.get(), 1);
    }

    #[test]
    fn inc_and_dec_on_the_owner_never_touch_shared() {
        const REFERENCES: usize = 1 << 20;
        let rc = RefCount::new();
        for _ in 1..REFERENCES {
            rc.inc();
        }
        assert_eq!(rc.get(), REFERENCES);
        for _ in 1..REFERENCES {
            assert!(!rc.dec());
        }
        assert_eq!(raw(&rc).2, INIT);
        assert_eq!(rc.get(), 1);
        assert!(rc.dec());
    }

    #[test]
    fn inc_by_on_the_owner() {
        let rc = RefCount::new();
        rc.inc_by(1 << 20);
        assert_eq!(rc.get(), (1 << 20) + 1);
    }

    #[test]
    fn an_immortal_count_never_moves() {
        let rc = RefCount::new();
        assert!(!rc.is_immortal());
        rc.make_immortal();
        assert!(rc.is_immortal());
        assert_eq!(rc.get(), IMMORTAL_COUNT);
        let before = raw(&rc);
        rc.inc();
        rc.inc_by(1000);
        assert!(rc.safe_inc());
        for _ in 0..1000 {
            assert!(!rc.dec());
        }
        assert_eq!(raw(&rc), before);
        rc.make_immortal();
        assert_eq!(rc.get(), IMMORTAL_COUNT);
    }

    #[test]
    fn interning_implies_immortality_but_not_the_other_way() {
        let immortal = RefCount::new();
        immortal.make_immortal();
        assert!(immortal.is_immortal());
        assert!(!immortal.is_leaked());

        let interned = RefCount::new();
        interned.leak();
        assert!(interned.is_leaked());
        assert!(interned.is_immortal());
        interned.inc();
        assert!(!interned.dec());
        assert!(!interned.dec());
        assert_eq!(interned.get(), IMMORTAL_COUNT);
    }

    #[test]
    fn published_bit_survives_refcount_traffic() {
        let rc = RefCount::new();
        assert!(!rc.is_published());
        rc.mark_published();
        assert!(rc.is_published());
        rc.inc();
        assert!(rc.is_published());
        assert!(!rc.dec());
        assert!(rc.safe_inc());
        assert!(!rc.dec());
        assert!(rc.dec());
        assert!(rc.is_published());
    }

    #[test]
    fn the_owner_merges_when_another_thread_holds_a_reference() {
        let rc = RefCount::new();
        std::thread::scope(|s| {
            s.spawn(|| rc.inc()).join().unwrap();
        });
        assert_eq!(rc.get(), 2);
        // The owner lets go of its reference: `shared` still holds one, so the
        // count merges instead of reaching zero.
        assert!(!rc.dec());
        let (tid, local, shared) = raw(&rc);
        assert_eq!((tid, local), (UNOWNED, 0));
        assert_eq!(shared & STATE_MASK, MERGED);
        assert_eq!(rc.get(), 1);
        // Merged: every thread now uses `shared`, including the old owner.
        rc.inc();
        assert!(!rc.dec());
        let last = std::thread::scope(|s| s.spawn(|| rc.dec()).join().unwrap());
        assert!(last);
    }

    #[test]
    fn a_release_by_a_non_owner_is_queued_and_merged_by_the_owner() {
        let rc = RefCount::new();
        rc.inc(); // the reference another thread will release
        let freed = std::thread::scope(|s| s.spawn(|| rc.dec()).join().unwrap());
        assert!(!freed);
        assert_eq!(raw(&rc).2 & STATE_MASK, QUEUED);
        assert_eq!(rc.get(), 2, "the queue holds the released reference");
        let mut deallocated = Vec::new();
        merge_queued_objects(|p| deallocated.push(p));
        assert!(deallocated.is_empty());
        assert_eq!(rc.get(), 1);
        assert_eq!(raw(&rc).0, UNOWNED);
        assert!(rc.dec());
    }

    #[test]
    fn the_queue_deallocates_when_it_held_the_last_reference() {
        let rc = RefCount::new();
        let freed = std::thread::scope(|s| s.spawn(|| rc.dec()).join().unwrap());
        assert!(!freed);
        let mut deallocated = Vec::new();
        merge_queued_objects(|p| deallocated.push(p));
        assert_eq!(deallocated, [&rc as *const RefCount]);
    }

    #[test]
    fn an_exited_owners_objects_are_merged_by_the_releasing_thread() {
        // The object outlives the thread that created it.
        let rc = std::thread::spawn(|| {
            let rc = RefCount::new();
            rc.inc();
            exit_current_thread(|_| unreachable!());
            // After exiting, the thread's own release goes to `shared`.
            assert!(!rc.dec());
            rc
        })
        .join()
        .unwrap();
        assert_eq!(rc.get(), 1);
        // `shared` is zero and the owner is gone: this thread merges.
        assert!(rc.dec());
    }

    #[test]
    fn safe_inc_from_another_thread_needs_maybe_weakref() {
        let rc = RefCount::new();
        let ok = std::thread::scope(|s| s.spawn(|| rc.safe_inc()).join().unwrap());
        assert!(
            !ok,
            "an unmarked unmerged object looks dead to other threads"
        );
        rc.set_maybe_weakref();
        let ok = std::thread::scope(|s| s.spawn(|| rc.safe_inc()).join().unwrap());
        assert!(ok);
        assert_eq!(rc.get(), 2);
        // The owner's release now merges through the CAS and leaves one.
        assert!(!rc.dec());
        assert_eq!(rc.get(), 1);
        assert!(rc.dec());
        // Dead: nobody can revive it.
        assert!(!rc.safe_inc());
    }

    /// Many threads pass references to one object back and forth; the object
    /// must be reported dead exactly once, after every reference is gone.
    #[test]
    fn concurrent_traffic_frees_exactly_once() {
        const THREADS: usize = 8;
        const ROUNDS: usize = 20_000;
        for _ in 0..20 {
            let rc = RefCount::new();
            rc.set_maybe_weakref();
            rc.inc_by(THREADS);
            let barrier = Barrier::new(THREADS);
            let frees: usize = std::thread::scope(|s| {
                let handles: Vec<_> = (0..THREADS)
                    .map(|_| {
                        s.spawn(|| {
                            barrier.wait();
                            let mut frees = 0;
                            for _ in 0..ROUNDS {
                                rc.inc();
                                if rc.safe_inc() {
                                    frees += rc.dec() as usize;
                                }
                                frees += rc.dec() as usize;
                            }
                            // Release the reference handed to this thread.
                            frees += rc.dec() as usize;
                            let mut queued = 0;
                            merge_queued_objects(|_| queued += 1);
                            frees + queued
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).sum()
            });
            // The creator still holds its original reference.
            let mut queued = 0;
            merge_queued_objects(|_| queued += 1);
            let last = rc.dec() as usize;
            let mut after = 0;
            merge_queued_objects(|_| after += 1);
            assert_eq!(frees + queued + last + after, 1);
        }
    }
}
