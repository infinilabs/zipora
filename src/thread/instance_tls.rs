//! Instance-Specific Thread-Local Storage
//!
//! Advanced TLS management system providing O(1) access to thread-local data
//! associated with specific object instances. Based on matrix storage for
//! efficient lookup and automatic cleanup.

use crate::error::{Result, ZiporaError};
use std::any::{Any, TypeId};
use std::cell::{RefCell, UnsafeCell};
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

/// Default matrix dimensions for TLS storage
const DEFAULT_ROWS: usize = 256;
const DEFAULT_COLS: usize = 256;

/// Instance-specific thread-local storage with matrix-based O(1) access
pub struct InstanceTls<T, const ROWS: usize = DEFAULT_ROWS, const COLS: usize = DEFAULT_COLS>
where
    T: Send + Sync + 'static,
{
    id: u32,
    /// Which incarnation of `id` this is; see [`TlsRow`].
    generation: u32,
    _phantom: PhantomData<T>,
    _cleanup: Arc<CleanupHandle<T, ROWS, COLS>>,
}

/// A cell of the per-thread matrix, allocated lazily on the heap inside a boxed row slice of length `COLS`.
///
/// Each cell carries the generation of the instance that wrote it. Ids are
/// recycled when an instance is dropped, but the dropping thread can only
/// reach its own matrix, so every other thread still holds the dead
/// instance's value in the cell the recycled id now maps to. Tagging the
/// value makes those leftovers invisible without sweeping anything.
///
/// `readers` counts active in-place `T::clone()` shared borrows of `value` on this cell.
/// While `readers > 0`, `value` is only accessed via shared references `&Option<(u32, T)>`;
/// any re-entrant `set` or `remove` on the same cell records its update in
/// `TlsMatrix::deferred` to be applied once `readers` returns to `0`.
struct TlsCell<T> {
    readers: std::cell::Cell<u32>,
    value: UnsafeCell<Option<(u32, T)>>,
}

impl<T> TlsCell<T> {
    #[inline(always)]
    const fn new() -> Self {
        Self {
            readers: std::cell::Cell::new(0),
            value: UnsafeCell::new(None),
        }
    }
}

type DeferredCellWrite<T> = (*const TlsCell<T>, Option<(u32, T)>);

/// Matrix storage for thread-local data (`Vec` of lazy heap-allocated row slices
/// so large `ROWS`/`COLS` never allocate on the stack or invalidate cell pointers).
struct TlsMatrix<T, const ROWS: usize, const COLS: usize> {
    rows: UnsafeCell<Vec<Option<*mut [TlsCell<T>]>>>,
    deferred: UnsafeCell<Vec<DeferredCellWrite<T>>>,
}

struct CellReadGuard<T, const ROWS: usize, const COLS: usize> {
    matrix_ptr: *const TlsMatrix<T, ROWS, COLS>,
    cell_ptr: *const TlsCell<T>,
}

impl<T, const ROWS: usize, const COLS: usize> Drop for CellReadGuard<T, ROWS, COLS> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: `cell_ptr` and `matrix_ptr` are owned by the current thread and remain valid
        // for the lifetime of the thread's `TlsMatrix`.
        unsafe {
            let r = (*self.cell_ptr).readers.get() - 1;
            (*self.cell_ptr).readers.set(r);
            if r == 0 && !(&*(*self.matrix_ptr).deferred.get()).is_empty() {
                (&*self.matrix_ptr).apply_deferred(self.cell_ptr);
            }
        }
    }
}

// SAFETY: Caller must ensure `matrix_ptr` and `cell_ptr` are valid on the current thread and
// that `(*cell_ptr).value` currently contains `Some((generation, _))`.
#[inline(always)]
unsafe fn clone_cell_in_place<T: Clone, const ROWS: usize, const COLS: usize>(
    matrix_ptr: *const TlsMatrix<T, ROWS, COLS>,
    cell_ptr: *const TlsCell<T>,
) -> T {
    // SAFETY: `cell_ptr` is valid on the current thread.
    let r = unsafe { (*cell_ptr).readers.get() };
    // SAFETY: `cell_ptr` is valid on the current thread.
    unsafe { (*cell_ptr).readers.set(r + 1) };
    let guard = CellReadGuard {
        matrix_ptr,
        cell_ptr,
    };
    let cloned = {
        // SAFETY: While `readers > 0`, no mutable reference to `(*cell_ptr).value` is created
        // (any re-entrant write to this cell is queued in `matrix.deferred`), so this shared
        // borrow `&T` remains valid throughout `val_ref.clone()`, including across re-entrant
        // `get()` / `get_value()` shared reads on the same cell.
        let val_ref: &T = unsafe {
            match (&*(*cell_ptr).value.get()).as_ref() {
                Some((_, v)) => v,
                None => unreachable!(),
            }
        };
        val_ref.clone()
    };
    drop(guard);
    cloned
}

// SAFETY: `drop_fn` is only called on `ptr` created by `Box::into_raw` for the matching `TlsMatrix`.
struct MatrixEntry {
    ptr: *mut (),
    drop_fn: unsafe fn(*mut ()),
}

/// Per-thread table of [`TlsMatrix`] instances with a 1-entry MRU lookup cache
/// and re-entrancy-safe destruction at thread exit.
struct TlsMatrices {
    last: std::cell::Cell<Option<(TypeId, *mut ())>>,
    map: RefCell<HashMap<TypeId, MatrixEntry>>,
}

impl TlsMatrices {
    fn new() -> Self {
        Self {
            last: std::cell::Cell::new(None),
            map: RefCell::new(HashMap::new()),
        }
    }

    #[inline(always)]
    fn get_or_init_matrix<T: 'static, const ROWS: usize, const COLS: usize>(
        &self,
    ) -> Option<*const TlsMatrix<T, ROWS, COLS>> {
        let type_id = TypeId::of::<TlsMatrix<T, ROWS, COLS>>();
        if let Some((cached_id, cached_ptr)) = self.last.get()
            && cached_id == type_id
        {
            return Some(cached_ptr as *const TlsMatrix<T, ROWS, COLS>);
        }
        self.get_or_init_matrix_slow::<T, ROWS, COLS>(type_id)
    }

    #[cold]
    #[inline(never)]
    fn get_or_init_matrix_slow<T: 'static, const ROWS: usize, const COLS: usize>(
        &self,
        type_id: TypeId,
    ) -> Option<*const TlsMatrix<T, ROWS, COLS>> {
        unsafe fn drop_matrix<T, const ROWS: usize, const COLS: usize>(ptr: *mut ()) {
            // SAFETY: `ptr` was allocated via `Box::into_raw(Box::new(TlsMatrix::<T, ROWS, COLS>::new()))`.
            unsafe {
                drop(Box::from_raw(ptr as *mut TlsMatrix<T, ROWS, COLS>));
            }
        }

        let mut map = self.map.try_borrow_mut().ok()?;
        let entry = map.entry(type_id).or_insert_with(|| {
            let boxed = Box::new(TlsMatrix::<T, ROWS, COLS>::new());
            MatrixEntry {
                ptr: Box::into_raw(boxed) as *mut (),
                drop_fn: drop_matrix::<T, ROWS, COLS>,
            }
        });
        let ptr = entry.ptr;
        drop(map);
        self.last.set(Some((type_id, ptr)));
        Some(ptr as *const TlsMatrix<T, ROWS, COLS>)
    }

    #[inline(always)]
    fn lookup_matrix<T: 'static, const ROWS: usize, const COLS: usize>(
        &self,
    ) -> Option<*const TlsMatrix<T, ROWS, COLS>> {
        let type_id = TypeId::of::<TlsMatrix<T, ROWS, COLS>>();
        if let Some((cached_id, cached_ptr)) = self.last.get()
            && cached_id == type_id
        {
            return Some(cached_ptr as *const TlsMatrix<T, ROWS, COLS>);
        }
        let map = self.map.try_borrow().ok()?;
        let ptr = map.get(&type_id)?.ptr;
        drop(map);
        self.last.set(Some((type_id, ptr)));
        Some(ptr as *const TlsMatrix<T, ROWS, COLS>)
    }
}

impl Drop for TlsMatrices {
    fn drop(&mut self) {
        // Drain entries one at a time outside `self.map.borrow_mut()` so that if a
        // stored value's `Drop` accesses any `InstanceTls` during thread exit,
        // `self.map` is not borrowed and remaining matrices are still valid.
        loop {
            self.last.set(None);
            let next_entry = {
                let Ok(mut map) = self.map.try_borrow_mut() else {
                    break;
                };
                let next_key = map.keys().next().copied();
                next_key.and_then(|k| map.remove(&k))
            };
            match next_entry {
                Some(entry) => {
                    // SAFETY: `entry.ptr` was created by `Box::into_raw` for the concrete
                    // `TlsMatrix` type paired with `entry.drop_fn` and was just removed from
                    // `self.map` after clearing `self.last`.
                    unsafe { (entry.drop_fn)(entry.ptr) };
                }
                None => break,
            }
        }
    }
}

/// Global state for managing TLS instance IDs
struct GlobalTlsState<T, const ROWS: usize, const COLS: usize>
where
    T: Send + Sync + 'static,
{
    /// Free list of recycled IDs, each with the generation to hand out next
    free_ids: VecDeque<(u32, u32)>,
    /// Next available ID
    next_id: u32,
    /// Cleanup handles for automatic deallocation
    cleanup_handles: Vec<Weak<CleanupHandle<T, ROWS, COLS>>>,
}

/// Cleanup handle for automatic resource deallocation
struct CleanupHandle<T, const ROWS: usize, const COLS: usize>
where
    T: Send + Sync + 'static,
{
    id: u32,
    generation: u32,
    _phantom: PhantomData<T>,
}

/// Per-`(T, ROWS, COLS)` id allocators, keyed by type.
///
/// One module-level static. `allocate_id` and `CleanupHandle::drop` used to
/// declare a function-local `static GLOBAL_REGISTRY` each, which are two
/// distinct statics: drop pushed freed ids into a map nobody allocated from,
/// so ids were never recycled and the allocator ran off the end of the
/// matrix after ROWS * COLS instances over the process lifetime.
type InstanceRegistry = RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>;

static GLOBAL_REGISTRY: OnceLock<InstanceRegistry> = OnceLock::new();

fn global_registry() -> &'static InstanceRegistry {
    GLOBAL_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

// Thread-local storage for matrices
thread_local! {
    static TLS_MATRICES: TlsMatrices = TlsMatrices::new();
}

impl<T, const ROWS: usize, const COLS: usize> InstanceTls<T, ROWS, COLS>
where
    T: Send + Sync + Default + Clone + 'static,
{
    /// Validate that `ROWS * COLS` is non-zero, does not overflow `usize`, and fits in `u32`.
    #[inline]
    fn validate_dimensions() -> Result<usize> {
        let total_cells = ROWS
            .checked_mul(COLS)
            .ok_or_else(|| ZiporaError::invalid_parameter("Matrix dimensions overflow usize"))?;
        if total_cells == 0 {
            return Err(ZiporaError::invalid_parameter(
                "Matrix dimensions cannot be zero",
            ));
        }
        if total_cells > u32::MAX as usize {
            return Err(ZiporaError::invalid_parameter(
                "Matrix dimensions exceed u32::MAX",
            ));
        }
        Ok(total_cells)
    }

    /// Create a new instance-specific TLS
    pub fn new() -> Result<Self> {
        Self::validate_dimensions()?;

        let (id, generation) = Self::allocate_id()?;
        let cleanup = Arc::new(CleanupHandle {
            id,
            generation,
            _phantom: PhantomData,
        });

        Ok(Self {
            id,
            generation,
            _phantom: PhantomData,
            _cleanup: cleanup,
        })
    }

    /// Get copy of thread-local instance data.
    ///
    /// If `T::clone()` re-entrantly calls [`Self::set`] or [`Self::remove`] on the
    /// same `InstanceTls` instance on the current thread, that mutation is deferred
    /// until the outermost active `clone()` on the cell completes (`readers == 0`).
    #[inline]
    pub fn get(&self) -> T
    where
        T: Clone,
    {
        let (row, col) = self.get_indices();
        let generation = self.generation;

        TLS_MATRICES
            .try_with(|tls| {
                let Some(matrix_ptr) = tls.get_or_init_matrix::<T, ROWS, COLS>() else {
                    return T::default();
                };
                // SAFETY: `matrix_ptr` is owned by the current thread's `TLS_MATRICES`, and
                // `ensure_cell` returns a stable heap pointer into the row's boxed slice.
                let cell_ptr = unsafe { (&*matrix_ptr).ensure_cell(row, col) };

                // SAFETY: If `readers > 0`, `value` is only read via shared reference; if
                // `readers == 0`, no reference to `value` exists on the stack.
                let is_hit = unsafe {
                    (&*(*cell_ptr).value.get())
                        .as_ref()
                        .is_some_and(|(stored, _)| *stored == generation)
                };
                if is_hit {
                    // SAFETY: `cell_ptr` contains `Some((generation, _))` and `matrix_ptr` is valid.
                    return unsafe { clone_cell_in_place(matrix_ptr, cell_ptr) };
                }

                // Cold miss or stale-generation path: evict any stale value (when no reader is
                // active on this cell) before invoking `T::default()` / `T::clone()`.
                // SAFETY: `readers == 0` is checked before forming `&mut *(*cell_ptr).value.get()`,
                // ensuring no shared reference into `(*cell_ptr).value` is live.
                let stale = unsafe {
                    if (*cell_ptr).readers.get() == 0 {
                        (&mut *(*cell_ptr).value.get()).take()
                    } else {
                        None
                    }
                };
                drop(stale);

                let default_val = T::default();
                let cloned = default_val.clone();
                // SAFETY: `cell_ptr` remains valid on the current thread. `readers == 0` is
                // checked before forming `&mut *(*cell_ptr).value.get()`, and `default_val` is
                // stored only if no re-entrant call during `T::default()` / `T::clone()`
                // populated or borrowed the cell.
                let displaced = unsafe {
                    if (*cell_ptr).readers.get() == 0 {
                        let slot = &mut *(*cell_ptr).value.get();
                        if slot.is_none() {
                            *slot = Some((generation, default_val));
                            None
                        } else {
                            Some(default_val)
                        }
                    } else {
                        Some(default_val)
                    }
                };
                drop(displaced);
                cloned
            })
            .unwrap_or_else(|_| T::default())
    }

    /// Get copy of thread-local instance data if it exists.
    ///
    /// If `T::clone()` re-entrantly calls [`Self::set`] or [`Self::remove`] on the
    /// same `InstanceTls` instance on the current thread, that mutation is deferred
    /// until the outermost active `clone()` on the cell completes (`readers == 0`).
    #[inline]
    pub fn get_value(&self) -> Option<T>
    where
        T: Clone,
    {
        let (row, col) = self.get_indices();
        let generation = self.generation;

        TLS_MATRICES
            .try_with(|tls| {
                let matrix_ptr = tls.lookup_matrix::<T, ROWS, COLS>()?;
                // SAFETY: `matrix_ptr` is valid on the current thread.
                let cell_ptr = unsafe { (&*matrix_ptr).lookup_cell(row, col)? };
                // SAFETY: Shared inspection of `(*cell_ptr).value` on the current thread.
                let matches = unsafe {
                    (&*(*cell_ptr).value.get())
                        .as_ref()
                        .is_some_and(|(stored, _)| *stored == generation)
                };
                if !matches {
                    return None;
                }
                // SAFETY: `cell_ptr` contains `Some((generation, _))` and `matrix_ptr` is valid.
                Some(unsafe { clone_cell_in_place(matrix_ptr, cell_ptr) })
            })
            .ok()
            .flatten()
    }

    /// Try to get value copy without creating default value
    #[inline]
    pub fn try_get(&self) -> Option<T>
    where
        T: Clone,
    {
        self.get_value()
    }

    /// Set the thread-local value.
    ///
    /// If called re-entrantly from inside `T::clone()` on the same `InstanceTls`
    /// instance and thread, the write is queued and takes effect as soon as the
    /// outermost `clone()` on that cell finishes.
    pub fn set(&self, value: T) {
        let (row, col) = self.get_indices();
        let generation = self.generation;

        let evicted = TLS_MATRICES
            .try_with(|tls| {
                let matrix_ptr = tls.get_or_init_matrix::<T, ROWS, COLS>()?;
                // SAFETY: `matrix_ptr` is valid on the current thread.
                let cell_ptr = unsafe { (&*matrix_ptr).ensure_cell(row, col) };
                // SAFETY: If `readers == 0`, no shared borrow of `(*cell_ptr).value` is active;
                // otherwise queue the update in `matrix.deferred` until the active clone finishes.
                unsafe {
                    if (*cell_ptr).readers.get() == 0 {
                        (&mut *(*cell_ptr).value.get()).replace((generation, value))
                    } else {
                        (&*matrix_ptr).defer_write(cell_ptr, Some((generation, value)));
                        None
                    }
                }
            })
            .ok()
            .flatten();
        drop(evicted);
    }

    /// Remove the thread-local value.
    ///
    /// If called re-entrantly from inside `T::clone()` on the same `InstanceTls`
    /// instance and thread, clearing the cell is deferred until the outermost
    /// `clone()` on that cell finishes and this call returns `None`.
    pub fn remove(&self) -> Option<T> {
        let (row, col) = self.get_indices();
        let generation = self.generation;

        TLS_MATRICES
            .try_with(|tls| {
                let matrix_ptr = tls.lookup_matrix::<T, ROWS, COLS>()?;
                // SAFETY: `matrix_ptr` is valid on the current thread.
                let cell_ptr = unsafe { (&*matrix_ptr).lookup_cell(row, col)? };
                // SAFETY: If `readers == 0`, no shared borrow of `(*cell_ptr).value` is active;
                // otherwise defer clearing the cell until the active clone finishes.
                unsafe {
                    if (*cell_ptr).readers.get() == 0 {
                        let slot = &mut *(*cell_ptr).value.get();
                        if slot.as_ref().is_some_and(|(stored, _)| *stored == generation) {
                            slot.take().map(|(_, v)| v)
                        } else {
                            None
                        }
                    } else {
                        (&*matrix_ptr).defer_write(cell_ptr, None);
                        None
                    }
                }
            })
            .ok()
            .flatten()
    }

    /// Get the unique instance ID
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Calculate matrix indices from ID
    #[inline]
    fn get_indices(&self) -> (usize, usize) {
        let id = self.id as usize;
        (id / COLS, id % COLS)
    }

    /// Allocate a unique instance ID and the generation that goes with it
    fn allocate_id() -> Result<(u32, u32)> {
        let total_cells = Self::validate_dimensions()?;
        let registry = global_registry();

        // Try read lock first to see if we already have this type
        let type_id = TypeId::of::<(T, [(); ROWS], [(); COLS])>(); // Use a unique type ID with const generics

        {
            let read_guard = registry.read().map_err(|_| {
                ZiporaError::system_error("Failed to acquire TLS registry read lock")
            })?;

            if let Some(state_box) = read_guard.get(&type_id)
                && let Some(state_mutex) =
                    state_box.downcast_ref::<Mutex<GlobalTlsState<T, ROWS, COLS>>>()
            {
                let mut state = state_mutex
                    .lock()
                    .map_err(|_| ZiporaError::system_error("Failed to acquire TLS state lock"))?;

                // Clean up dead handles
                state
                    .cleanup_handles
                    .retain(|handle| handle.strong_count() > 0);

                // Try to reuse a free ID
                if let Some((id, generation)) = state.free_ids.pop_front() {
                    return Ok((id, generation));
                } else {
                    let id = state.next_id;
                    if id as usize >= total_cells {
                        return Err(ZiporaError::resource_exhausted(format!(
                            "Too many TLS instances: max {}",
                            total_cells
                        )));
                    }
                    state.next_id = state.next_id.checked_add(1).ok_or_else(|| {
                        ZiporaError::resource_exhausted("TLS instance ID counter exhausted")
                    })?;
                    return Ok((id, 0));
                }
            }
        }

        // Need write lock to insert new type
        let mut write_guard = registry
            .write()
            .map_err(|_| ZiporaError::system_error("Failed to acquire TLS registry write lock"))?;

        let state = write_guard.entry(type_id).or_insert_with(|| {
            Box::new(Mutex::new(GlobalTlsState::<T, ROWS, COLS> {
                free_ids: VecDeque::new(),
                next_id: 0,
                cleanup_handles: Vec::new(),
            }))
        });

        let state_mutex = state
            .downcast_ref::<Mutex<GlobalTlsState<T, ROWS, COLS>>>()
            .ok_or_else(|| ZiporaError::system_error("Type downcast failed"))?;

        let mut state = state_mutex
            .lock()
            .map_err(|_| ZiporaError::system_error("Failed to acquire TLS state lock"))?;

        // Try to reuse a free ID
        if let Some((id, generation)) = state.free_ids.pop_front() {
            Ok((id, generation))
        } else {
            let id = state.next_id;
            if id as usize >= total_cells {
                return Err(ZiporaError::resource_exhausted(format!(
                    "Too many TLS instances: max {}",
                    total_cells
                )));
            }
            state.next_id = state.next_id.checked_add(1).ok_or_else(|| {
                ZiporaError::resource_exhausted("TLS instance ID counter exhausted")
            })?;
            Ok((id, 0))
        }
    }
}

impl<T, const ROWS: usize, const COLS: usize> Clone for InstanceTls<T, ROWS, COLS>
where
    T: Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            generation: self.generation,
            _phantom: PhantomData,
            _cleanup: Arc::clone(&self._cleanup),
        }
    }
}

impl<T, const ROWS: usize, const COLS: usize> TlsMatrix<T, ROWS, COLS> {
    /// Create a new TLS matrix with no rows allocated yet.
    fn new() -> Self {
        Self {
            rows: UnsafeCell::new(Vec::new()),
            deferred: UnsafeCell::new(Vec::new()),
        }
    }

    #[cold]
    #[inline(never)]
    fn alloc_row(&self, row: usize) -> *mut [TlsCell<T>] {
        let mut cells = Vec::with_capacity(COLS);
        cells.resize_with(COLS, TlsCell::new);
        let ptr = Box::into_raw(cells.into_boxed_slice());
        // SAFETY: `self.rows` is only accessed on the owning thread and is not borrowed across calls.
        unsafe {
            let rows = &mut *self.rows.get();
            if row >= rows.len() {
                rows.resize(row + 1, None);
            }
            rows[row] = Some(ptr);
        }
        ptr
    }

    /// Return a raw pointer to the `(row, col)` cell, allocating the row on demand.
    #[inline(always)]
    fn ensure_cell(&self, row: usize, col: usize) -> *const TlsCell<T> {
        // SAFETY: `self.rows` is only accessed on the owning thread; borrow ends before `alloc_row`.
        let existing_row = unsafe { (&*self.rows.get()).get(row).copied().flatten() };
        let row_ptr = match existing_row {
            Some(ptr) => ptr,
            None => self.alloc_row(row),
        };
        // SAFETY: `row_ptr` points to a heap-allocated `[TlsCell<T>]` of length `COLS`
        // and `col < COLS` is guaranteed by `get_indices()`.
        unsafe { &raw const (*row_ptr)[col] }
    }

    /// Return a raw pointer to the `(row, col)` cell if the row has been allocated.
    #[inline(always)]
    fn lookup_cell(&self, row: usize, col: usize) -> Option<*const TlsCell<T>> {
        // SAFETY: `self.rows` is only accessed on the owning thread.
        let row_ptr = unsafe { (&*self.rows.get()).get(row).copied().flatten()? };
        // SAFETY: `row_ptr` points to a heap-allocated `[TlsCell<T>]` of length `COLS`
        // and `col < COLS` is guaranteed by `get_indices()`.
        Some(unsafe { &raw const (*row_ptr)[col] })
    }

    #[cold]
    #[inline(never)]
    fn defer_write(&self, cell_ptr: *const TlsCell<T>, next: Option<(u32, T)>) {
        // SAFETY: `self.deferred` is only accessed on the owning thread; any displaced
        // pending value is extracted before dropping outside the borrow.
        let replaced = unsafe {
            let deferred = &mut *self.deferred.get();
            if let Some(slot) = deferred.iter_mut().find(|(c, _)| *c == cell_ptr) {
                std::mem::replace(&mut slot.1, next)
            } else {
                deferred.push((cell_ptr, next));
                None
            }
        };
        drop(replaced);
    }

    #[cold]
    #[inline(never)]
    fn apply_deferred(&self, cell_ptr: *const TlsCell<T>) {
        // SAFETY: Called only when `(*cell_ptr).readers.get() == 0` after all shared borrows
        // of `(*cell_ptr).value` have ended.
        let pending = unsafe {
            let deferred = &mut *self.deferred.get();
            deferred
                .iter()
                .position(|(c, _)| *c == cell_ptr)
                .map(|idx| deferred.remove(idx).1)
        };
        if let Some(next_val) = pending {
            // SAFETY: `readers == 0` on `cell_ptr`; extract old value to drop outside borrow.
            let old = unsafe { std::mem::replace(&mut *(*cell_ptr).value.get(), next_val) };
            drop(old);
        }
    }
}

impl<T, const ROWS: usize, const COLS: usize> Drop for TlsMatrix<T, ROWS, COLS> {
    fn drop(&mut self) {
        self.deferred.get_mut().clear();
        for row_opt in self.rows.get_mut() {
            if let Some(row_ptr) = row_opt.take() {
                // SAFETY: `row_ptr` was allocated via `Box::into_raw` in `alloc_row` and is
                // owned exclusively by this `TlsMatrix`.
                unsafe {
                    drop(Box::from_raw(row_ptr));
                }
            }
        }
    }
}

// SAFETY: TlsMatrix owns heap-allocated row slices of `TlsCell<T>` and is only
// accessed from its owning thread; transferring ownership across threads is safe when `T: Send`.
unsafe impl<T: Send, const ROWS: usize, const COLS: usize> Send for TlsMatrix<T, ROWS, COLS> {}

impl<T: Send + Sync + 'static, const ROWS: usize, const COLS: usize> Drop
    for CleanupHandle<T, ROWS, COLS>
{
    fn drop(&mut self) {
        // Eagerly clear the dropping thread's cell so resources held by `T` on the
        // current thread are released immediately rather than lingering until ID reuse.
        if let Some(row) = (self.id as usize).checked_div(COLS) {
            let col = (self.id as usize) % COLS;
            let generation = self.generation;
            let removed = TLS_MATRICES
                .try_with(|tls| {
                    let matrix_ptr = tls.lookup_matrix::<T, ROWS, COLS>()?;
                    // SAFETY: `matrix_ptr` is valid on the current thread.
                    let cell_ptr = unsafe { (&*matrix_ptr).lookup_cell(row, col)? };
                    // SAFETY: `cell_ptr` is valid on the current thread, and `readers == 0` is
                    // checked before forming `&mut *(*cell_ptr).value.get()`, ensuring no shared
                    // reference into `(*cell_ptr).value` is live on the call stack.
                    unsafe {
                        if (*cell_ptr).readers.get() == 0 {
                            let slot = &mut *(*cell_ptr).value.get();
                            if slot.as_ref().is_some_and(|(stored, _)| *stored == generation) {
                                slot.take().map(|(_, v)| v)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                })
                .ok()
                .flatten();
            drop(removed);
        }

        // Return ID to free list - use the same registry as allocation.
        // Only recycle the ID if `generation + 1` does not overflow u32::MAX;
        // otherwise retire the ID permanently to prevent generation wrap-around aliasing.
        if let Some(next_gen) = self.generation.checked_add(1) {
            let registry = global_registry();
            if let Ok(read_guard) = registry.read() {
                let type_id = TypeId::of::<(T, [(); ROWS], [(); COLS])>();
                if let Some(state_box) = read_guard.get(&type_id)
                    && let Some(state_mutex) =
                        state_box.downcast_ref::<Mutex<GlobalTlsState<T, ROWS, COLS>>>()
                    && let Ok(mut state) = state_mutex.lock()
                {
                    state.free_ids.push_back((self.id, next_gen));
                }
            }
        }
    }
}

/// Owner-based TLS that associates thread-local data with specific owners
pub struct OwnerTls<T, O>
where
    T: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    instances: HashMap<*const O, InstanceTls<T>>,
    _phantom: PhantomData<O>,
}

impl<T, O> Default for OwnerTls<T, O>
where
    T: Send + Sync + Default + Clone + 'static,
    O: Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, O> OwnerTls<T, O>
where
    T: Send + Sync + Default + Clone + 'static,
    O: Send + Sync + 'static,
{
    /// Create a new owner-based TLS
    pub fn new() -> Self {
        Self {
            instances: HashMap::new(),
            _phantom: PhantomData,
        }
    }

    /// Get or create TLS for a specific owner
    pub fn get_or_create(&mut self, owner: &O) -> Result<T> {
        let owner_ptr = owner as *const O;

        match self.instances.entry(owner_ptr) {
            std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.get().get()),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let instance = InstanceTls::new()?;
                Ok(entry.insert(instance).get())
            }
        }
    }

    /// Get TLS for owner if it exists
    #[inline]
    pub fn get(&self, owner: &O) -> Option<T> {
        let owner_ptr = owner as *const O;
        self.instances.get(&owner_ptr)?.get_value()
    }

    /// Remove TLS for owner
    pub fn remove(&mut self, owner: &O) -> Option<InstanceTls<T>> {
        let owner_ptr = owner as *const O;
        self.instances.remove(&owner_ptr)
    }

    /// Get number of registered owners
    #[inline]
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

/// Thread-local storage pool for managing multiple TLS instances
pub struct TlsPool<T, const POOL_SIZE: usize = 64>
where
    T: Send + Sync + 'static,
{
    pool: Vec<InstanceTls<T>>,
    next_slot: AtomicU32,
}

impl<T, const POOL_SIZE: usize> TlsPool<T, POOL_SIZE>
where
    T: Send + Sync + Default + Clone + 'static,
{
    /// Create a new TLS pool
    pub fn new() -> Result<Self> {
        if POOL_SIZE == 0 {
            return Err(ZiporaError::invalid_parameter(
                "TLS pool size cannot be zero",
            ));
        }

        let mut pool = Vec::with_capacity(POOL_SIZE);
        for _ in 0..POOL_SIZE {
            pool.push(InstanceTls::new()?);
        }

        Ok(Self {
            pool,
            next_slot: AtomicU32::new(0),
        })
    }

    /// Get the next available TLS instance (round-robin)
    pub fn get_next(&self) -> T {
        // ORDERING: Relaxed is sufficient because next_slot is an independent round-robin
        // counter and does not synchronize any shared state across threads.
        let slot = self.next_slot.fetch_add(1, Ordering::Relaxed) as usize % POOL_SIZE;
        self.pool[slot].get()
    }

    /// Get TLS instance by slot index
    pub fn get_slot(&self, slot: usize) -> Option<T> {
        self.pool.get(slot).map(InstanceTls::get)
    }

    /// Get number of slots in pool
    #[inline]
    pub fn len(&self) -> usize {
        POOL_SIZE
    }

    /// Returns `true` if the pool has no slots (always false for a valid pool)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[derive(Debug, Default, Clone, PartialEq)]
    struct TestData {
        value: u32,
        name: String,
    }

    // TestData is automatically Send + Sync since all its fields are Send + Sync

    #[test]
    fn test_instance_tls_basic() {
        let tls = InstanceTls::<TestData>::new().unwrap();

        // Default value should be created
        assert_eq!(tls.get().value, 0);
        assert_eq!(tls.get().name, "");

        // Set new value
        let test_data = TestData {
            value: 42,
            name: "test".to_string(),
        };
        tls.set(test_data);

        // Value should be retrievable
        let retrieved = tls.get();
        assert_eq!(retrieved.value, 42);
        assert_eq!(retrieved.name, "test");
    }

    #[test]
    fn test_instance_tls_multiple_threads() {
        let tls = Arc::new(InstanceTls::<TestData>::new().unwrap());

        let handles: Vec<_> = (0..5)
            .map(|i| {
                let tls = Arc::clone(&tls);
                thread::spawn(move || {
                    // Each thread should get its own instance
                    let test_data = TestData {
                        value: i * 10,
                        name: format!("thread_{}", i),
                    };
                    tls.set(test_data);

                    thread::sleep(Duration::from_millis(10));

                    // Value should remain unchanged
                    let retrieved = tls.get();
                    assert_eq!(retrieved.value, i * 10);
                    assert_eq!(retrieved.name, format!("thread_{}", i));
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_instance_tls_set_remove() {
        let tls = InstanceTls::<TestData>::new().unwrap();

        // Initially should be None using try_get (which doesn't create a default)
        assert!(tls.try_get().is_none());

        // Set a value
        let test_data = TestData {
            value: 100,
            name: "test_set".to_string(),
        };
        tls.set(test_data);

        // Should now exist
        assert!(tls.try_get().is_some());
        let retrieved = tls.try_get().unwrap();
        assert_eq!(retrieved.value, 100);

        // Remove the value
        let removed = tls.remove().unwrap();
        assert_eq!(removed.value, 100);
        assert_eq!(removed.name, "test_set");

        // Should be None again
        assert!(tls.try_get().is_none());
    }

    #[test]
    fn test_owner_tls() {
        struct Owner {
            #[allow(dead_code)] // distinct non-ZST identity (used by address)
            id: u32,
        }

        let owner1 = Owner { id: 1 };
        let owner2 = Owner { id: 2 };

        let mut owner_tls = OwnerTls::<TestData, Owner>::new();

        // Each owner should have separate TLS
        let mut data1 = owner_tls.get_or_create(&owner1).unwrap();
        data1.value = 11;
        owner_tls.get_or_create(&owner1).unwrap(); // This creates it with the modified value

        let mut data2 = owner_tls.get_or_create(&owner2).unwrap();
        data2.value = 22;
        owner_tls.get_or_create(&owner2).unwrap(); // This creates it with the modified value

        // Note: Since we're returning by value, we need to use set() to persist changes
        // Let's modify this test to use set() properly
        let _test_data1 = TestData {
            value: 11,
            name: "owner1".to_string(),
        };
        let _test_data2 = TestData {
            value: 22,
            name: "owner2".to_string(),
        };

        // We need to access the instances to call set() on them
        // For this test, let's verify that different owners get different instances
        let retrieved1 = owner_tls.get_or_create(&owner1).unwrap();
        let retrieved2 = owner_tls.get_or_create(&owner2).unwrap();

        // They should both start with default values (0, "")
        assert_eq!(retrieved1.value, 0);
        assert_eq!(retrieved2.value, 0);

        // Remove owner1
        assert!(owner_tls.remove(&owner1).is_some());
        assert!(owner_tls.get(&owner1).is_none());
        assert_eq!(owner_tls.get(&owner2).unwrap().value, 0); // Still default value
    }

    #[test]
    fn test_tls_pool() {
        let pool = TlsPool::<TestData, 4>::new().unwrap();

        // Test round-robin access - since we return by value,
        // we can't modify the stored values directly
        // Instead, let's just test that we get different values in round-robin
        for _i in 0..8 {
            let _tls = pool.get_next(); // Just test that it doesn't panic
        }

        // Test that slots return default values
        assert_eq!(pool.get_slot(0).unwrap().value, 0);
        assert_eq!(pool.get_slot(1).unwrap().value, 0);
        assert_eq!(pool.get_slot(2).unwrap().value, 0);
        assert_eq!(pool.get_slot(3).unwrap().value, 0);
    }

    #[test]
    fn test_tls_id_management() {
        let tls1 = InstanceTls::<TestData>::new().unwrap();
        let tls2 = InstanceTls::<TestData>::new().unwrap();

        let id1 = tls1.id();
        let id2 = tls2.id();

        // IDs should be different
        assert_ne!(id1, id2);

        // Drop first instance
        drop(tls1);

        // Create new instance - might reuse ID
        let tls3 = InstanceTls::<TestData>::new().unwrap();
        let _id3 = tls3.id();

        // This tests the ID recycling mechanism
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct RecycleProbe(u32);

    #[derive(Debug, Default, Clone, PartialEq)]
    struct StaleProbe(u32);

    #[test]
    fn test_instance_ids_are_recycled_after_drop() {
        // Regression: allocate_id and CleanupHandle::drop each declared their
        // own function-local `static GLOBAL_REGISTRY`. Those are two distinct
        // statics, so drop pushed freed ids into a map nobody allocated from
        // and no id was ever reused. A 2x2 matrix ran out after the fourth
        // instance ever created; with the default 256x256 matrix any process
        // that creates more than 65,536 instances over its lifetime fails
        // permanently.
        for i in 0..16 {
            let tls = InstanceTls::<RecycleProbe, 2, 2>::new()
                .unwrap_or_else(|e| panic!("instance {i} could not allocate an id: {e}"));
            drop(tls);
        }
    }

    #[test]
    fn test_recycled_id_does_not_expose_the_previous_instances_value() {
        // Recycling ids means a new instance lands on the matrix cell its
        // predecessor used. Drop can only reach the dropping thread's matrix,
        // so the value has to be tagged with a generation instead.
        let first = InstanceTls::<StaleProbe, 2, 2>::new().unwrap();
        let id = first.id();
        first.set(StaleProbe(42));
        assert_eq!(first.get_value(), Some(StaleProbe(42)));
        drop(first);

        let second = InstanceTls::<StaleProbe, 2, 2>::new().unwrap();
        assert_eq!(
            second.id(),
            id,
            "this test is vacuous unless the id is recycled"
        );
        assert_eq!(
            second.get_value(),
            None,
            "the dead instance's value leaked into its successor"
        );
        assert_eq!(second.try_get(), None);
        assert_eq!(second.remove(), None);
        assert_eq!(second.get(), StaleProbe::default());
    }

    #[test]
    fn test_matrix_dimensions() {
        // Test custom matrix dimensions
        let tls = InstanceTls::<TestData, 4, 4>::new().unwrap();
        let test_data = TestData {
            value: 99,
            name: "test".to_string(),
        };
        tls.set(test_data);
        assert_eq!(tls.get().value, 99);
    }

    #[test]
    fn test_c10_f1_drop_clears_current_thread_and_reentrant_set_does_not_panic() {
        #[derive(Clone, Default)]
        struct DropTracker(#[allow(dead_code)] Option<Arc<()>>);

        let token = Arc::new(());
        let tls = InstanceTls::<DropTracker, 2, 2>::new().unwrap();
        tls.set(DropTracker(Some(Arc::clone(&token))));
        assert_eq!(Arc::strong_count(&token), 2);

        drop(tls);
        assert_eq!(
            Arc::strong_count(&token),
            1,
            "dropping InstanceTls must immediately drop the current thread's stored value"
        );

        // Also verify that overwriting a value whose Drop reads/writes another InstanceTls
        // does not panic with RefCell BorrowMutError.
        let inner_tls = Arc::new(InstanceTls::<u32, 2, 2>::new().unwrap());
        #[derive(Clone, Default)]
        struct ReentrantDrop(Option<Arc<InstanceTls<u32, 2, 2>>>);
        impl Drop for ReentrantDrop {
            fn drop(&mut self) {
                if let Some(ref tls) = self.0 {
                    tls.set(77);
                }
            }
        }

        let outer_tls = InstanceTls::<ReentrantDrop, 2, 2>::new().unwrap();
        outer_tls.set(ReentrantDrop(Some(Arc::clone(&inner_tls))));
        outer_tls.set(ReentrantDrop(None));
        assert_eq!(inner_tls.get(), 77);
    }

    #[test]
    fn test_c10_f2_zero_pool_and_overflow_dimensions_return_err() {
        assert!(
            TlsPool::<TestData, 0>::new().is_err(),
            "TlsPool with POOL_SIZE == 0 must return Err"
        );
        assert!(
            InstanceTls::<TestData, 65536, 65536>::new().is_err(),
            "InstanceTls with ROWS * COLS > u32::MAX must return Err"
        );
    }

    #[test]
    fn test_s25_r7_sequential_70k_churn_on_default_dimensions_no_leak() {
        use std::sync::atomic::AtomicI64;

        static LIVE_A1: AtomicI64 = AtomicI64::new(0);
        #[derive(Debug)]
        struct CountedA1(u64);
        impl Default for CountedA1 {
            fn default() -> Self {
                LIVE_A1.fetch_add(1, Ordering::Relaxed);
                Self(0)
            }
        }
        impl Clone for CountedA1 {
            fn clone(&self) -> Self {
                LIVE_A1.fetch_add(1, Ordering::Relaxed);
                Self(self.0)
            }
        }
        impl Drop for CountedA1 {
            fn drop(&mut self) {
                LIVE_A1.fetch_sub(1, Ordering::Relaxed);
            }
        }

        // Default matrix is 256x256 = 65,536 cells. Hold `cap` live instances
        // simultaneously so every ID in `0..cap` is allocated (and a `cap + 1`th
        // allocation on the full matrix returns `Err`), then drop all `cap` and
        // allocate `extra` recycled instances (70,000 total in non-Miri builds).
        let (cap, extra) = if cfg!(miri) {
            (32, 32)
        } else {
            (256 * 256, 70_000 - 256 * 256)
        };
        let mut live = Vec::with_capacity(cap);
        for i in 0..cap {
            let tls = InstanceTls::<CountedA1>::new().unwrap();
            tls.set(CountedA1({
                LIVE_A1.fetch_add(1, Ordering::Relaxed);
                i as u64
            }));
            assert_eq!(tls.get().0, i as u64);
            live.push(tls);
        }
        if !cfg!(miri) {
            assert!(
                InstanceTls::<CountedA1>::new().is_err(),
                "65,537th live instance on a full 256x256 matrix must return Err"
            );
        }
        drop(live);
        assert_eq!(
            LIVE_A1.load(Ordering::Relaxed),
            0,
            "dropping all 65,536 live instances must drop their stored values"
        );
        for i in 0..extra {
            let val = (cap + i) as u64;
            let tls = InstanceTls::<CountedA1>::new().unwrap();
            tls.set(CountedA1({
                LIVE_A1.fetch_add(1, Ordering::Relaxed);
                val
            }));
            assert_eq!(tls.get().0, val);
            drop(tls);
        }
        assert_eq!(
            LIVE_A1.load(Ordering::Relaxed),
            0,
            "all stored values must be dropped after sequential churn"
        );
    }

    #[test]
    fn test_s25_r7_parallel_80k_churn_across_threads() {
        use std::sync::atomic::AtomicI64;

        static LIVE_A2: AtomicI64 = AtomicI64::new(0);
        #[derive(Debug)]
        struct CountedA2(u64);
        impl Default for CountedA2 {
            fn default() -> Self {
                LIVE_A2.fetch_add(1, Ordering::Relaxed);
                Self(0)
            }
        }
        impl Clone for CountedA2 {
            fn clone(&self) -> Self {
                LIVE_A2.fetch_add(1, Ordering::Relaxed);
                Self(self.0)
            }
        }
        impl Drop for CountedA2 {
            fn drop(&mut self) {
                LIVE_A2.fetch_sub(1, Ordering::Relaxed);
            }
        }

        let (threads, per_thread) = if cfg!(miri) { (2, 16) } else { (8, 10_000) };
        let mut handles = Vec::with_capacity(threads);
        for t in 0..threads {
            handles.push(thread::spawn(move || {
                for i in 0..per_thread {
                    let tls = InstanceTls::<CountedA2, 32, 32>::new().unwrap();
                    let expected = (t * 100_000 + i) as u64;
                    tls.set(CountedA2({
                        LIVE_A2.fetch_add(1, Ordering::Relaxed);
                        expected
                    }));
                    assert_eq!(tls.get().0, expected);
                    drop(tls);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(
            LIVE_A2.load(Ordering::Relaxed),
            0,
            "all worker-thread values must be dropped after thread exit"
        );
    }

    #[test]
    fn test_s25_r5_reentrant_drop_at_thread_exit_does_not_abort() {
        // S25-R5 regression: when a worker thread exits without dropping the
        // `InstanceTls` handle first, `TLS_MATRICES` drops its stored values
        // during thread-local destruction. If a stored value's `Drop` touches
        // another `InstanceTls` (via `get`, `try_get`, `set`, or `remove`),
        // `TLS_MATRICES.with` previously panicked inside `Drop` and aborted the
        // process.
        use std::sync::atomic::AtomicUsize;

        static DROPPED_ON_EXIT: AtomicUsize = AtomicUsize::new(0);

        let target_tls = Arc::new(InstanceTls::<u64, 4, 4>::new().unwrap());

        #[derive(Clone, Default)]
        struct ExitToucher(Option<Arc<InstanceTls<u64, 4, 4>>>);
        impl Drop for ExitToucher {
            fn drop(&mut self) {
                if let Some(ref other) = self.0 {
                    let _ = other.try_get();
                    let _ = other.get();
                    other.set(999);
                    let _ = other.remove();
                    DROPPED_ON_EXIT.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        let holder_tls = Arc::new(InstanceTls::<ExitToucher, 4, 4>::new().unwrap());
        {
            let holder = Arc::clone(&holder_tls);
            let target = Arc::clone(&target_tls);
            thread::spawn(move || {
                target.set(123);
                holder.set(ExitToucher(Some(target)));
                // Thread exits while `holder_tls` and `target_tls` remain alive on main thread.
            })
            .join()
            .unwrap();
        }
        assert_eq!(DROPPED_ON_EXIT.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_s25_r9_reentrant_clone_and_large_rows_do_not_panic_or_overflow_stack() {
        // S25-R9 (a): `T::clone()` touching another (or the same) `InstanceTls`
        // during `get()` / `get_value()` / `try_get()` must not panic with
        // `RefCell already borrowed`.
        let side_tls = Arc::new(InstanceTls::<u32, 4, 4>::new().unwrap());

        #[derive(Default)]
        struct CloneToucher(Option<Arc<InstanceTls<u32, 4, 4>>>, u32);
        impl Clone for CloneToucher {
            fn clone(&self) -> Self {
                if let Some(ref side) = self.0 {
                    let prev = side.get();
                    side.set(prev + 1);
                }
                Self(self.0.clone(), self.1)
            }
        }

        let main_tls = InstanceTls::<CloneToucher, 4, 4>::new().unwrap();
        main_tls.set(CloneToucher(Some(Arc::clone(&side_tls)), 42));
        assert_eq!(main_tls.get().1, 42);
        assert_eq!(main_tls.get_value().unwrap().1, 42);
        assert_eq!(main_tls.try_get().unwrap().1, 42);
        assert_eq!(side_tls.get(), 3);

        // S25-R9 (b): Large `ROWS` (`4_000_000 x 1`) must allocate rows lazily
        // on the heap rather than constructing `[Option<Box<...>>; ROWS]` on
        // the stack (which previously overflowed the 8 MiB thread stack).
        let large_tls = InstanceTls::<u64, 4_000_000, 1>::new().unwrap();
        large_tls.set(0xDEAD_BEEF);
        assert_eq!(large_tls.get(), 0xDEAD_BEEF);
    }

    #[test]
    fn test_s26_r2_reentrant_same_instance_get_and_get_value_in_clone() {
        use std::cell::Cell;

        thread_local! {
            static ENTERED_GET: Cell<bool> = const { Cell::new(false) };
            static ENTERED_GET_VALUE: Cell<bool> = const { Cell::new(false) };
        }

        #[derive(Default)]
        struct SelfGet(u64, Option<Arc<InstanceTls<SelfGet, 4, 4>>>);
        impl Clone for SelfGet {
            fn clone(&self) -> Self {
                if let Some(ref tls) = self.1
                    && !ENTERED_GET.with(|c| c.replace(true))
                {
                    let inner = tls.get();
                    assert_eq!(
                        inner.0, self.0,
                        "re-entrant get() on the same InstanceTls must read the stored value"
                    );
                }
                Self(self.0, self.1.clone())
            }
        }

        let tls1 = Arc::new(InstanceTls::<SelfGet, 4, 4>::new().unwrap());
        tls1.set(SelfGet(5, Some(Arc::clone(&tls1))));
        let out1 = tls1.get();
        assert_eq!(out1.0, 5);
        assert_eq!(
            tls1.get_value().map(|v| v.0),
            Some(5),
            "re-entrant get() during Clone must not overwrite stored value with T::default()"
        );

        #[derive(Default)]
        struct SelfGetValue(u64, Option<Arc<InstanceTls<SelfGetValue, 4, 4>>>);
        impl Clone for SelfGetValue {
            fn clone(&self) -> Self {
                if let Some(ref tls) = self.1
                    && !ENTERED_GET_VALUE.with(|c| c.replace(true))
                {
                    let inner = tls.get_value().expect(
                        "re-entrant get_value() on the same InstanceTls must return Some",
                    );
                    assert_eq!(inner.0, self.0);
                }
                Self(self.0, self.1.clone())
            }
        }

        let tls2 = Arc::new(InstanceTls::<SelfGetValue, 4, 4>::new().unwrap());
        tls2.set(SelfGetValue(5, Some(Arc::clone(&tls2))));
        let out2 = tls2.get_value().map(|v| v.0);
        assert_eq!(out2, Some(5));
        assert_eq!(tls2.get_value().map(|v| v.0), Some(5));

        // Re-entrant `set()` on the same instance during `Clone` must defer the
        // write until the active `Clone` finishes without aliasing `&self`.
        #[derive(Default)]
        struct SelfSet(u64, Option<Arc<InstanceTls<SelfSet, 4, 4>>>);
        impl Clone for SelfSet {
            fn clone(&self) -> Self {
                if let Some(ref tls) = self.1 {
                    tls.set(SelfSet(99, None));
                }
                Self(self.0, None)
            }
        }

        let tls3 = Arc::new(InstanceTls::<SelfSet, 4, 4>::new().unwrap());
        tls3.set(SelfSet(7, Some(Arc::clone(&tls3))));
        assert_eq!(tls3.get().0, 7);
        assert_eq!(tls3.get_value().map(|v| v.0), Some(99));
    }

    #[test]
    fn test_s26_r2_panicking_clone_preserves_stored_value() {
        use std::sync::atomic::AtomicBool;

        static SHOULD_PANIC: AtomicBool = AtomicBool::new(true);

        #[derive(Default, Debug, PartialEq, Eq)]
        struct FlakyClone(u64);
        impl Clone for FlakyClone {
            fn clone(&self) -> Self {
                if SHOULD_PANIC.load(Ordering::Relaxed) {
                    panic!("simulated panic in FlakyClone::clone");
                }
                Self(self.0)
            }
        }

        let tls = InstanceTls::<FlakyClone, 4, 4>::new().unwrap();
        tls.set(FlakyClone(77));

        let res_get = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tls.get()));
        assert!(res_get.is_err());
        let res_get_val =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tls.get_value()));
        assert!(res_get_val.is_err());

        SHOULD_PANIC.store(false, Ordering::Relaxed);
        assert_eq!(
            tls.get_value(),
            Some(FlakyClone(77)),
            "panicking Clone must leave the stored value intact in the cell"
        );
        assert_eq!(tls.get(), FlakyClone(77));
    }
}
