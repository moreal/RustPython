//! Reference counting for `PyObject`.
//!
//! A build with the `threading` feature uses biased reference counting (see
//! [`biased`]): the thread that created an object counts its own references
//! with plain loads and stores, and only other threads pay for atomic
//! read-modify-writes. A build without it has a single thread, so a single
//! plain counter word (see [`single`]) is enough.

#[cfg(feature = "threading")]
mod biased;
#[cfg(not(feature = "threading"))]
mod single;

#[cfg(feature = "threading")]
pub use biased::{
    RefCount, after_fork_child, current_owner_id, exit_current_thread, has_queued_objects,
    merge_queued_objects, merge_queued_objects_of, others_have_queued_objects, reinit_after_fork,
};
#[cfg(not(feature = "threading"))]
pub use single::RefCount;

// Deferred Drop Infrastructure
//
// This mechanism allows untrack_object() calls to be deferred until after
// the GC collection phase completes, preventing deadlocks that occur when
// clear (pop_edges) triggers object destruction while holding the tracked_objects lock.

#[cfg(feature = "std")]
use core::cell::{Cell, RefCell};

#[cfg(feature = "std")]
thread_local! {
    /// Flag indicating if we're inside a deferred drop context.
    /// When true, drop operations should defer untrack calls.
    static IN_DEFERRED_CONTEXT: Cell<bool> = const { Cell::new(false) };

    /// Queue of deferred untrack operations.
    /// No Send bound needed - this is thread-local and only accessed from the same thread.
    static DEFERRED_QUEUE: RefCell<Vec<Box<dyn FnOnce()>>> = const { RefCell::new(Vec::new()) };
}

#[cfg(feature = "std")]
struct DeferredDropGuard {
    was_in_context: bool,
}

#[cfg(feature = "std")]
impl Drop for DeferredDropGuard {
    fn drop(&mut self) {
        IN_DEFERRED_CONTEXT.with(|in_ctx| {
            in_ctx.set(self.was_in_context);
        });
        // Only flush if we're the outermost context and not already panicking
        // (flushing during unwinding risks double-panic → process abort).
        if !self.was_in_context && !std::thread::panicking() {
            flush_deferred_drops();
        }
    }
}

/// Execute a function within a deferred drop context.
/// Any calls to `try_defer_drop` within this context will be queued
/// and executed when the context exits (even on panic).
#[cfg(feature = "std")]
#[inline]
pub fn with_deferred_drops<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = IN_DEFERRED_CONTEXT.with(|in_ctx| {
        let was_in_context = in_ctx.get();
        in_ctx.set(true);
        DeferredDropGuard { was_in_context }
    });
    f()
}

/// Try to defer a drop-related operation.
/// If inside a deferred context, the operation is queued.
/// Otherwise, it executes immediately.
#[cfg(feature = "std")]
#[inline]
pub fn try_defer_drop<F>(f: F)
where
    F: FnOnce() + 'static,
{
    let should_defer = IN_DEFERRED_CONTEXT.with(|in_ctx| in_ctx.get());

    if should_defer {
        DEFERRED_QUEUE.with(|q| {
            q.borrow_mut().push(Box::new(f));
        });
    } else {
        f();
    }
}

/// Flush all deferred drop operations.
/// This is automatically called when exiting a deferred context.
#[cfg(feature = "std")]
#[inline]
pub fn flush_deferred_drops() {
    DEFERRED_QUEUE.with(|q| {
        // Take all queued operations
        let ops: Vec<_> = q.borrow_mut().drain(..).collect();
        // Execute them outside the borrow
        for op in ops {
            op();
        }
    });
}
