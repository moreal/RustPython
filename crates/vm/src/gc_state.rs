//! Garbage Collection State and Algorithm
//!
//! This module implements CPython-compatible generational garbage collection
//! for RustPython.
//!
//! Tracking is kept deliberately cheap: objects are recorded in a single
//! pointer-keyed table mapping each tracked object to its generation, using a
//! trivial pointer hash. Objects are removed from the table when they are
//! deallocated (see `default_dealloc`), so the table never contains dangling
//! pointers.

use crate::common::lock::PyMutex;
use crate::{PyObject, PyObjectRef};
use core::hash::BuildHasherDefault;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

bitflags::bitflags! {
    /// GC debug flags (see Include/internal/pycore_gc.h)
    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    pub struct GcDebugFlags: u32 {
        /// Print collection statistics
        const STATS         = 1 << 0;
        /// Print collectable objects
        const COLLECTABLE   = 1 << 1;
        /// Print uncollectable objects
        const UNCOLLECTABLE = 1 << 2;
        /// Save all garbage in gc.garbage
        const SAVEALL       = 1 << 5;
        /// DEBUG_COLLECTABLE | DEBUG_UNCOLLECTABLE | DEBUG_SAVEALL
        const LEAK = Self::COLLECTABLE.bits() | Self::UNCOLLECTABLE.bits() | Self::SAVEALL.bits();
    }
}

/// Statistics for a single generation (gc_generation_stats)
#[derive(Debug, Default, Clone, Copy)]
pub struct GcStats {
    pub collections: usize,
    pub collected: usize,
    pub uncollectable: usize,
}

/// A single GC generation
pub struct GcGeneration {
    /// Number of objects in this generation
    count: AtomicUsize,
    /// Threshold for triggering collection
    threshold: AtomicU32,
    /// Collection statistics
    stats: PyMutex<GcStats>,
}

impl GcGeneration {
    pub const fn new(threshold: u32) -> Self {
        Self {
            count: AtomicUsize::new(0),
            threshold: AtomicU32::new(threshold),
            stats: PyMutex::new(GcStats {
                collections: 0,
                collected: 0,
                uncollectable: 0,
            }),
        }
    }

    pub fn count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    pub fn threshold(&self) -> u32 {
        self.threshold.load(Ordering::Relaxed)
    }

    pub fn set_threshold(&self, value: u32) {
        self.threshold.store(value, Ordering::Relaxed);
    }

    pub fn stats(&self) -> GcStats {
        let guard = self.stats.lock();
        GcStats {
            collections: guard.collections,
            collected: guard.collected,
            uncollectable: guard.uncollectable,
        }
    }

    pub fn update_stats(&self, collected: usize, uncollectable: usize) {
        let mut guard = self.stats.lock();
        guard.collections += 1;
        guard.collected += collected;
        guard.uncollectable += uncollectable;
    }

    /// Decrement the count, saturating at zero.
    fn dec_count(&self) {
        let _ = self
            .count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| c.checked_sub(1));
    }
}

/// Wrapper for raw pointer to make it Send + Sync
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GcObjectPtr(NonNull<PyObject>);

// SAFETY: We only use this for tracking objects, and proper synchronization is used
unsafe impl Send for GcObjectPtr {}
unsafe impl Sync for GcObjectPtr {}

/// A trivial hasher for object pointers. Tracking runs on every allocation of
/// a GC-eligible object, so hashing must be as cheap as possible; a single
/// multiply disperses the (already mostly unique) address bits well enough for
/// a hash table, unlike the default SipHash which costs far more per key.
#[derive(Default)]
struct PtrHasher(u64);

impl core::hash::Hasher for PtrHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(b)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
    }

    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.0 = n.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

/// Generation index stored for frozen (permanent) objects.
const PERMANENT_GEN: u8 = 3;

/// Map from tracked object to the generation it currently lives in
/// (0-2, or `PERMANENT_GEN` for frozen objects).
type TrackedMap = HashMap<GcObjectPtr, u8, BuildHasherDefault<PtrHasher>>;

/// Global GC state
pub struct GcState {
    /// 3 generations (0 = youngest, 2 = oldest)
    pub generations: [GcGeneration; 3],
    /// Permanent generation (frozen objects)
    pub permanent: GcGeneration,
    /// GC enabled flag
    pub enabled: AtomicBool,
    /// All tracked objects and the generation each lives in
    tracked: RwLock<TrackedMap>,
    /// Debug flags
    pub debug: AtomicU32,
    /// gc.garbage list (uncollectable objects with __del__)
    pub garbage: PyMutex<Vec<PyObjectRef>>,
    /// gc.callbacks list
    pub callbacks: PyMutex<Vec<PyObjectRef>>,
    /// Mutex for collection (prevents concurrent collections).
    /// Used by collect_inner when the actual collection algorithm is enabled.
    #[allow(dead_code)]
    collecting: Mutex<()>,
}

// SAFETY: All fields are either inherently Send/Sync (atomics, RwLock, Mutex) or protected by PyMutex.
// PyMutex<Vec<PyObjectRef>> is safe to share/send across threads because access is synchronized.
// PyObjectRef itself is Send, and interior mutability is guarded by the mutex.
unsafe impl Send for GcState {}
unsafe impl Sync for GcState {}

impl Default for GcState {
    fn default() -> Self {
        Self::new()
    }
}

impl GcState {
    pub fn new() -> Self {
        Self {
            generations: [
                GcGeneration::new(2000), // young
                GcGeneration::new(10),   // old[0]
                GcGeneration::new(0),    // old[1]
            ],
            permanent: GcGeneration::new(0),
            enabled: AtomicBool::new(true),
            tracked: RwLock::new(TrackedMap::default()),
            debug: AtomicU32::new(0),
            garbage: PyMutex::new(Vec::new()),
            callbacks: PyMutex::new(Vec::new()),
            collecting: Mutex::new(()),
        }
    }

    /// Check if GC is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Enable GC
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    /// Disable GC
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    /// Get debug flags
    pub fn get_debug(&self) -> GcDebugFlags {
        GcDebugFlags::from_bits_truncate(self.debug.load(Ordering::Relaxed))
    }

    /// Set debug flags
    pub fn set_debug(&self, flags: GcDebugFlags) {
        self.debug.store(flags.bits(), Ordering::Relaxed);
    }

    /// Get thresholds for all generations
    pub fn get_threshold(&self) -> (u32, u32, u32) {
        (
            self.generations[0].threshold(),
            self.generations[1].threshold(),
            self.generations[2].threshold(),
        )
    }

    /// Set thresholds
    pub fn set_threshold(&self, t0: u32, t1: Option<u32>, t2: Option<u32>) {
        self.generations[0].set_threshold(t0);
        if let Some(t1) = t1 {
            self.generations[1].set_threshold(t1);
        }
        if let Some(t2) = t2 {
            self.generations[2].set_threshold(t2);
        }
    }

    /// Get counts for all generations
    pub fn get_count(&self) -> (usize, usize, usize) {
        (
            self.generations[0].count(),
            self.generations[1].count(),
            self.generations[2].count(),
        )
    }

    /// Get statistics for all generations
    pub fn get_stats(&self) -> [GcStats; 3] {
        [
            self.generations[0].stats(),
            self.generations[1].stats(),
            self.generations[2].stats(),
        ]
    }

    /// Track a new object (add to gen0)
    /// Called when IS_TRACE objects are created
    ///
    /// # Safety
    /// obj must be a valid pointer to a PyObject
    pub unsafe fn track_object(&self, obj: NonNull<PyObject>) {
        let gc_ptr = GcObjectPtr(obj);

        // _PyObject_GC_TRACK
        let obj_ref = unsafe { obj.as_ref() };
        obj_ref.set_gc_tracked();

        if let Ok(mut tracked) = self.tracked.write()
            && tracked.insert(gc_ptr, 0).is_none()
        {
            self.generations[0].count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Untrack an object (remove from GC lists)
    /// Called when objects are deallocated
    ///
    /// # Safety
    /// obj must be a valid pointer to a PyObject
    pub unsafe fn untrack_object(&self, obj: NonNull<PyObject>) {
        let gc_ptr = GcObjectPtr(obj);

        let removed = match self.tracked.write() {
            Ok(mut tracked) => tracked.remove(&gc_ptr),
            Err(_) => None,
        };
        match removed {
            Some(generation @ 0..=2) => self.generations[generation as usize].dec_count(),
            Some(_) => self.permanent.dec_count(),
            None => {}
        }
    }

    /// Get tracked objects (for gc.get_objects)
    /// If generation is None, returns all tracked objects.
    /// If generation is Some(n), returns objects in generation n only.
    ///
    /// Objects in the table are guaranteed to be alive: deallocation removes
    /// them under the same lock before their memory is freed. Objects that are
    /// mid-destruction (refcount 0) fail `try_to_owned` and are skipped.
    pub fn get_objects(&self, generation: Option<i32>) -> Vec<PyObjectRef> {
        let Ok(tracked) = self.tracked.read() else {
            return Vec::new();
        };
        tracked
            .iter()
            .filter_map(|(ptr, &obj_gen)| {
                if let Some(g) = generation
                    && obj_gen != g as u8
                {
                    return None;
                }
                let obj = unsafe { ptr.0.as_ref() };
                obj.try_to_owned()
            })
            .collect()
    }

    /// Check if automatic GC should run and run it if needed.
    /// Called after object allocation.
    /// Returns true if GC was run, false otherwise.
    pub fn maybe_collect(&self) -> bool {
        // _PyObject_GC_Alloc checks thresholds

        // Check gen0 threshold
        let count0 = self.generations[0].count.load(Ordering::Relaxed) as u32;
        let threshold0 = self.generations[0].threshold();
        if threshold0 > 0 && count0 >= threshold0 && self.is_enabled() {
            self.collect(0);
            return true;
        }

        false
    }

    /// Perform garbage collection on the given generation.
    /// Returns (collected_count, uncollectable_count).
    ///
    /// Currently a stub — the actual collection algorithm requires EBR
    /// and will be added in a follow-up.
    pub fn collect(&self, _generation: usize) -> (usize, usize) {
        // gc_collect_main
        // Reset gen0 count even though we're not actually collecting
        self.generations[0].count.store(0, Ordering::Relaxed);
        (0, 0)
    }

    /// Force collection even if GC is disabled (for manual gc.collect() calls).
    /// gc.collect() always runs regardless of gc.isenabled()
    /// Currently a stub.
    pub fn collect_force(&self, _generation: usize) -> (usize, usize) {
        // Reset gen0 count even though we're not actually collecting
        self.generations[0].count.store(0, Ordering::Relaxed);
        (0, 0)
    }

    /// Get count of frozen objects
    pub fn get_freeze_count(&self) -> usize {
        self.permanent.count()
    }

    /// Freeze all tracked objects (move to permanent generation)
    pub fn freeze(&self) {
        if let Ok(mut tracked) = self.tracked.write() {
            let mut frozen = 0usize;
            for obj_gen in tracked.values_mut() {
                if *obj_gen != PERMANENT_GEN {
                    *obj_gen = PERMANENT_GEN;
                    frozen += 1;
                }
            }
            for generation in &self.generations {
                generation.count.store(0, Ordering::Relaxed);
            }
            self.permanent.count.fetch_add(frozen, Ordering::Relaxed);
        }
    }

    /// Unfreeze all objects (move from permanent to gen2)
    pub fn unfreeze(&self) {
        if let Ok(mut tracked) = self.tracked.write() {
            let mut unfrozen = 0usize;
            for obj_gen in tracked.values_mut() {
                if *obj_gen == PERMANENT_GEN {
                    *obj_gen = 2;
                    unfrozen += 1;
                }
            }
            self.permanent.count.store(0, Ordering::Relaxed);
            self.generations[2]
                .count
                .fetch_add(unfrozen, Ordering::Relaxed);
        }
    }
}

use std::sync::OnceLock;

/// Global GC state instance
/// Using a static because GC needs to be accessible from object allocation/deallocation
static GC_STATE: OnceLock<GcState> = OnceLock::new();

/// Get a reference to the global GC state
pub fn gc_state() -> &'static GcState {
    GC_STATE.get_or_init(GcState::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gc_state_default() {
        let state = GcState::new();
        assert!(state.is_enabled());
        assert_eq!(state.get_debug(), GcDebugFlags::empty());
        assert_eq!(state.get_threshold(), (2000, 10, 0));
        assert_eq!(state.get_count(), (0, 0, 0));
    }

    #[test]
    fn test_gc_enable_disable() {
        let state = GcState::new();
        assert!(state.is_enabled());
        state.disable();
        assert!(!state.is_enabled());
        state.enable();
        assert!(state.is_enabled());
    }

    #[test]
    fn test_gc_threshold() {
        let state = GcState::new();
        state.set_threshold(100, Some(20), Some(30));
        assert_eq!(state.get_threshold(), (100, 20, 30));
    }

    #[test]
    fn test_gc_debug_flags() {
        let state = GcState::new();
        state.set_debug(GcDebugFlags::STATS | GcDebugFlags::COLLECTABLE);
        assert_eq!(
            state.get_debug(),
            GcDebugFlags::STATS | GcDebugFlags::COLLECTABLE
        );
    }
}
