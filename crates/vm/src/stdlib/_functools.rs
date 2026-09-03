pub(crate) use _functools::module_def;

#[pymodule]
mod _functools {
    use crate::{
        Py, PyObject, PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine,
        builtins::{
            PyBoundMethod, PyDict, PyDictRef, PyGenericAlias, PyTuple, PyType, PyTypeRef, object,
        },
        common::{
            hash::PyHash,
            lock::{PyMutex, PyRwLock},
        },
        dict_inner::DictKey,
        function::{FuncArgs, KwArgs, OptionalArg, OptionalOption, PySetterValue},
        object::AsObject,
        protocol::PyIter,
        pyclass,
        recursion::ReprGuard,
        types::{Callable, Constructor, GetDescriptor, Representable},
    };
    use rustpython_common::wtf8::Wtf8Buf;
    use std::collections::HashMap;

    #[derive(FromArgs)]
    struct ReduceArgs {
        function: PyObjectRef,
        iterator: PyIter,
        #[pyarg(any, optional, name = "initial")]
        initial: OptionalOption<PyObjectRef>,
    }

    #[pyfunction]
    fn reduce(args: ReduceArgs, vm: &VirtualMachine) -> PyResult {
        let ReduceArgs {
            function,
            iterator,
            initial,
        } = args;
        let mut iter = iterator.iter(vm)?;
        // OptionalOption distinguishes between:
        // - Missing: no argument provided → use first element from iterator
        // - Present(None): explicitly passed None → use None as initial value
        // - Present(Some(v)): passed a value → use that value
        let start_value = if let Some(val) = initial.into_option() {
            // initial was provided (could be None or Some value)
            val.unwrap_or_else(|| vm.ctx.none())
        } else {
            // initial was not provided at all
            iter.next().transpose()?.ok_or_else(|| {
                vm.new_type_error("reduce() of empty iterable with no initial value")
            })?
        };

        let mut accumulator = start_value;
        for next_obj in iter {
            accumulator = function.call((accumulator, next_obj?), vm)?
        }
        Ok(accumulator)
    }

    // Placeholder singleton for partial arguments
    // The singleton is stored as _instance on the type class
    #[pyattr]
    #[allow(non_snake_case)]
    fn Placeholder(vm: &VirtualMachine) -> PyObjectRef {
        let placeholder = PyPlaceholderType.into_pyobject(vm);
        // Store the singleton on the type class for slot_new to find
        let typ = placeholder.class();
        typ.set_attr(vm.ctx.intern_str("_instance"), placeholder.clone());
        placeholder
    }

    #[pyattr]
    #[pyclass(name = "_PlaceholderType", module = "functools")]
    #[derive(Debug, PyPayload)]
    pub(super) struct PyPlaceholderType;

    impl Constructor for PyPlaceholderType {
        type Args = FuncArgs;

        fn slot_new(cls: PyTypeRef, args: FuncArgs, vm: &VirtualMachine) -> PyResult {
            if !args.args.is_empty() || !args.kwargs.is_empty() {
                return Err(vm.new_type_error("_PlaceholderType takes no arguments"));
            }
            // Return the singleton stored on the type class
            if let Some(instance) = cls.get_attr(vm.ctx.intern_str("_instance")) {
                return Ok(instance);
            }
            // Fallback: create a new instance (shouldn't happen for base type after module init)
            Ok(Self.into_pyobject(vm))
        }

        fn py_new(_cls: &Py<PyType>, _args: Self::Args, _vm: &VirtualMachine) -> PyResult<Self> {
            // This is never called because we override slot_new
            Ok(Self)
        }
    }

    #[pyclass(with(Constructor, Representable))]
    impl PyPlaceholderType {
        #[pymethod]
        fn __reduce__(&self) -> &'static str {
            "Placeholder"
        }

        #[pymethod]
        fn __init_subclass__(_cls: PyTypeRef, vm: &VirtualMachine) -> PyResult<()> {
            Err(vm.new_type_error("cannot subclass '_PlaceholderType'"))
        }
    }

    impl Representable for PyPlaceholderType {
        #[inline]
        fn repr_str(_zelf: &Py<Self>, _vm: &VirtualMachine) -> PyResult<String> {
            Ok("Placeholder".to_owned())
        }
    }

    fn is_placeholder(obj: &PyObjectRef) -> bool {
        &*obj.class().name() == "_PlaceholderType"
    }

    fn count_placeholders(args: &[PyObjectRef]) -> usize {
        args.iter().filter(|a| is_placeholder(a)).count()
    }

    #[pyattr]
    #[pyclass(name = "partial", module = "functools")]
    #[derive(Debug, PyPayload)]
    pub(super) struct PyPartial {
        inner: PyRwLock<PyPartialInner>,
    }

    #[derive(Debug)]
    struct PyPartialInner {
        func: PyObjectRef,
        args: PyRef<PyTuple>,
        keywords: PyRef<PyDict>,
        phcount: usize,
    }

    #[pyclass(
        with(Constructor, Callable, GetDescriptor, Representable),
        flags(BASETYPE, HAS_DICT, HAS_WEAKREF)
    )]
    impl PyPartial {
        #[pygetset]
        fn func(&self) -> PyObjectRef {
            self.inner.read().func.clone()
        }

        #[pygetset]
        fn args(&self) -> PyRef<PyTuple> {
            self.inner.read().args.clone()
        }

        #[pygetset]
        fn keywords(&self) -> PyRef<PyDict> {
            self.inner.read().keywords.clone()
        }

        #[pygetset]
        fn __dict__(zelf: &Py<Self>, vm: &VirtualMachine) -> PyDictRef {
            zelf.as_object()
                .instance_dict()
                .map_or_else(|| vm.ctx.new_dict(), |d| d.get_or_insert(vm))
        }

        #[pygetset(setter)]
        fn set___dict__(
            zelf: &Py<Self>,
            value: PySetterValue,
            vm: &VirtualMachine,
        ) -> PyResult<()> {
            object::object_generic_set_dict(zelf.as_object().to_owned(), value, vm)
        }

        #[pymethod]
        fn __reduce__(zelf: &Py<Self>, vm: &VirtualMachine) -> PyObjectRef {
            let inner = zelf.inner.read();
            let partial_type = zelf.class();

            // Get __dict__ if it exists and is not empty
            let dict_obj = match zelf.as_object().dict() {
                Some(dict) if !dict.is_empty() => dict.into(),
                _ => vm.ctx.none(),
            };

            let state = vm.ctx.new_tuple(vec![
                inner.func.clone(),
                inner.args.clone().into(),
                inner.keywords.clone().into(),
                dict_obj,
            ]);
            vm.ctx
                .new_tuple(vec![
                    partial_type.to_owned().into(),
                    vm.ctx.new_tuple(vec![inner.func.clone()]).into(),
                    state.into(),
                ])
                .into()
        }

        #[pymethod]
        fn __setstate__(zelf: &Py<Self>, state: PyObjectRef, vm: &VirtualMachine) -> PyResult<()> {
            let state_tuple = state
                .downcast::<PyTuple>()
                .map_err(|_| vm.new_type_error("argument to __setstate__ must be a tuple"))?;

            if state_tuple.len() != 4 {
                return Err(vm.new_type_error(format!(
                    "expected 4 items in state, got {}",
                    state_tuple.len()
                )));
            }

            let func = &state_tuple[0];
            let args = &state_tuple[1];
            let kwds = &state_tuple[2];
            let dict = &state_tuple[3];

            if !func.is_callable() {
                return Err(vm.new_type_error("invalid partial state"));
            }

            // Validate that args is a tuple (or subclass)
            if !args.fast_isinstance(vm.ctx.types.tuple_type) {
                return Err(vm.new_type_error("invalid partial state"));
            }
            // Always convert to base tuple, even if it's a subclass
            let args_tuple = match args.clone().downcast::<PyTuple>() {
                Ok(tuple) if tuple.class().is(vm.ctx.types.tuple_type) => tuple,
                _ => {
                    // It's a tuple subclass, convert to base tuple
                    let elements: Vec<PyObjectRef> = args.try_to_value(vm)?;
                    vm.ctx.new_tuple(elements)
                }
            };

            let keywords_dict = if kwds.is(&vm.ctx.none) {
                vm.ctx.new_dict()
            } else {
                // Always convert to base dict, even if it's a subclass
                let dict = kwds
                    .clone()
                    .downcast::<PyDict>()
                    .map_err(|_| vm.new_type_error("invalid partial state"))?;
                if dict.class().is(vm.ctx.types.dict_type) {
                    // It's already a base dict
                    dict
                } else {
                    // It's a dict subclass, convert to base dict
                    let new_dict = vm.ctx.new_dict();
                    for (key, value) in dict {
                        new_dict.set_item(&*key, value, vm)?;
                    }
                    new_dict
                }
            };

            // Validate no trailing placeholders
            let args_slice = args_tuple.as_slice();
            if !args_slice.is_empty() && is_placeholder(args_slice.last().unwrap()) {
                return Err(vm.new_type_error("trailing Placeholders are not allowed"));
            }
            let phcount = count_placeholders(args_slice);

            // Actually update the state
            let mut inner = zelf.inner.write();
            inner.func = func.clone();
            // Handle args - use the already validated tuple
            inner.args = args_tuple;

            // Handle keywords - keep the original type
            inner.keywords = keywords_dict;
            inner.phcount = phcount;

            // Update __dict__ if provided
            let Some(instance_dict) = zelf.as_object().dict() else {
                return Ok(());
            };

            if dict.is(&vm.ctx.none) {
                // If dict is None, clear the instance dict
                instance_dict.clear();
                return Ok(());
            }

            let dict_obj = dict
                .clone()
                .downcast::<PyDict>()
                .map_err(|_| vm.new_type_error("invalid partial state"))?;

            // Clear existing dict and update with new values
            instance_dict.clear();
            for (key, value) in dict_obj {
                instance_dict.set_item(&*key, value, vm)?;
            }

            Ok(())
        }

        #[pyclassmethod]
        fn __class_getitem__(
            cls: PyTypeRef,
            args: PyObjectRef,
            vm: &VirtualMachine,
        ) -> PyResult<PyGenericAlias> {
            PyGenericAlias::from_args(cls, args, vm)
        }
    }

    impl Constructor for PyPartial {
        type Args = FuncArgs;

        fn py_new(
            _cls: &crate::Py<crate::builtins::PyType>,
            args: Self::Args,
            vm: &VirtualMachine,
        ) -> PyResult<Self> {
            let (func, args_slice) = args
                .args
                .split_first()
                .ok_or_else(|| vm.new_type_error("partial expected at least 1 argument, got 0"))?;

            if !func.is_callable() {
                return Err(vm.new_type_error("the first argument must be callable"));
            }

            // Check for placeholders in kwargs
            for (key, value) in &args.kwargs {
                if is_placeholder(value) {
                    return Err(vm.new_type_error(format!(
                        "Placeholder cannot be passed as a keyword argument to partial(). \
                         Did you mean partial(..., {key}=Placeholder, ...)(value)?"
                    )));
                }
            }

            // Handle nested partial objects
            let (final_func, final_args, final_keywords) =
                if let Some(partial) = func.downcast_ref::<Self>() {
                    let inner = partial.inner.read();
                    let stored_args = inner.args.as_slice();

                    // Merge placeholders: replace placeholders in stored_args with new args
                    let mut merged_args = Vec::with_capacity(stored_args.len() + args_slice.len());
                    let mut new_args_iter = args_slice.iter();

                    for stored_arg in stored_args {
                        if is_placeholder(stored_arg) {
                            // Replace placeholder with next new arg, or keep placeholder
                            if let Some(new_arg) = new_args_iter.next() {
                                merged_args.push(new_arg.clone());
                            } else {
                                merged_args.push(stored_arg.clone());
                            }
                        } else {
                            merged_args.push(stored_arg.clone());
                        }
                    }
                    // Append remaining new args
                    merged_args.extend(new_args_iter.cloned());

                    (inner.func.clone(), merged_args, inner.keywords.clone())
                } else {
                    (func.clone(), args_slice.to_vec(), vm.ctx.new_dict())
                };

            // Trailing placeholders are not allowed
            if !final_args.is_empty() && is_placeholder(final_args.last().unwrap()) {
                return Err(vm.new_type_error("trailing Placeholders are not allowed"));
            }

            let phcount = count_placeholders(&final_args);

            // Add new keywords
            for (key, value) in args.kwargs {
                final_keywords.set_item(vm.ctx.intern_str(key), value, vm)?;
            }

            Ok(Self {
                inner: PyRwLock::new(PyPartialInner {
                    func: final_func,
                    args: vm.ctx.new_tuple(final_args),
                    keywords: final_keywords,
                    phcount,
                }),
            })
        }
    }

    impl Callable for PyPartial {
        type Args = FuncArgs;

        fn call(zelf: &Py<Self>, args: FuncArgs, vm: &VirtualMachine) -> PyResult {
            // Clone and release lock before calling Python code to prevent deadlock
            let (func, stored_args, keywords, phcount) = {
                let inner = zelf.inner.read();
                (
                    inner.func.clone(),
                    inner.args.clone(),
                    inner.keywords.clone(),
                    inner.phcount,
                )
            };

            // Check if we have enough args to fill placeholders
            if phcount > 0 && args.args.len() < phcount {
                return Err(vm.new_type_error(format!(
                    "missing positional arguments in 'partial' call; expected at least {}, got {}",
                    phcount,
                    args.args.len()
                )));
            }

            // Build combined args, replacing placeholders
            let mut combined_args = Vec::with_capacity(stored_args.len() + args.args.len());
            let mut new_args_iter = args.args.iter();

            for stored_arg in stored_args.as_slice() {
                if is_placeholder(stored_arg) {
                    // Replace placeholder with next new arg
                    if let Some(new_arg) = new_args_iter.next() {
                        combined_args.push(new_arg.clone());
                    } else {
                        // This shouldn't happen if phcount check passed
                        combined_args.push(stored_arg.clone());
                    }
                } else {
                    combined_args.push(stored_arg.clone());
                }
            }
            // Append remaining new args
            combined_args.extend(new_args_iter.cloned());

            // Merge keywords from self.keywords and args.kwargs
            let mut final_kwargs = crate::function::KwArgsMap::default();

            // Add keywords from self.keywords
            for (key, value) in &*keywords {
                // `expect_str()` would panic on surrogate keys; keep them as WTF-8.
                let key_str = key
                    .downcast_ref::<crate::builtins::PyStr>()
                    .ok_or_else(|| vm.new_type_error("keywords must be strings"))?;
                final_kwargs.insert(key_str.as_wtf8().to_owned(), value);
            }

            // Add keywords from args.kwargs (these override self.keywords)
            for (key, value) in args.kwargs {
                final_kwargs.insert(key, value);
            }

            func.call(FuncArgs::new(combined_args, KwArgs::new(final_kwargs)), vm)
        }
    }

    impl GetDescriptor for PyPartial {
        fn descr_get(
            zelf: PyObjectRef,
            obj: Option<PyObjectRef>,
            _cls: Option<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult {
            let obj = match obj {
                Some(obj) if !vm.is_none(&obj) => obj,
                _ => return Ok(zelf),
            };
            Ok(PyBoundMethod::new(obj, zelf).into_ref(&vm.ctx).into())
        }
    }

    impl Representable for PyPartial {
        #[inline]
        fn repr_wtf8(zelf: &Py<Self>, vm: &VirtualMachine) -> PyResult<Wtf8Buf> {
            // Check for recursive repr
            let obj = zelf.as_object();
            if let Some(_guard) = ReprGuard::enter(vm, obj) {
                // Clone and release lock before calling Python code to prevent deadlock
                let (func, args, keywords) = {
                    let inner = zelf.inner.read();
                    (
                        inner.func.clone(),
                        inner.args.clone(),
                        inner.keywords.clone(),
                    )
                };

                let qualname = zelf.class().__qualname__(vm);
                let qualname_wtf8 = qualname
                    .downcast_ref::<crate::builtins::PyStr>()
                    .map_or_else(
                        || Wtf8Buf::from(zelf.class().name().to_owned()),
                        |s| s.as_wtf8().to_owned(),
                    );
                let module = zelf.class().__module__(vm);

                let mut result = Wtf8Buf::new();
                if let Ok(module_str) = module.downcast::<crate::builtins::PyStr>() {
                    let module_name = module_str.as_wtf8();
                    if module_name != "builtins" && !module_name.is_empty() {
                        result.push_wtf8(module_name);
                        result.push_char('.');
                    }
                }
                result.push_wtf8(&qualname_wtf8);
                result.push_char('(');
                result.push_wtf8(func.repr(vm)?.as_wtf8());

                for arg in args.as_slice() {
                    result.push_str(", ");
                    result.push_wtf8(arg.repr(vm)?.as_wtf8());
                }

                for (key, value) in &*keywords {
                    result.push_str(", ");
                    let key_str = if let Ok(s) = key.clone().downcast::<crate::builtins::PyStr>() {
                        s
                    } else {
                        key.str(vm)?
                    };
                    result.push_wtf8(key_str.as_wtf8());
                    result.push_char('=');
                    result.push_wtf8(value.repr(vm)?.as_wtf8());
                }

                result.push_char(')');
                Ok(result)
            } else {
                Ok(Wtf8Buf::from("..."))
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum LruMaxSize {
        NoCache,
        Unbounded,
        Bounded(usize),
    }

    #[derive(Debug)]
    struct LruNode {
        key: PyObjectRef,
        hash: PyHash,
        value: PyObjectRef,
        prev: Option<usize>,
        next: Option<usize>,
    }

    /// Cache storage backing the bounded/unbounded `maxsize` variants: a slab
    /// of nodes threaded into a doubly-linked recency list (`front` is least
    /// recently used, `back` is most recently used), indexed by a plain
    /// hash-bucket map so a lookup calls the key's `__hash__` exactly once.
    /// A node's slot is only ever reused in place by eviction (mirroring
    /// CPython's `lru_cache`), so `nodes` never grows a hole to track.
    #[derive(Debug, Default)]
    struct LruCacheState {
        buckets: HashMap<PyHash, Vec<usize>>,
        nodes: Vec<LruNode>,
        front: Option<usize>,
        back: Option<usize>,
        hits: usize,
        misses: usize,
    }

    impl LruCacheState {
        fn len(&self) -> usize {
            self.nodes.len()
        }

        /// Identity-only match within a hash bucket. Safe to call under the
        /// cache's lock: unlike `key_eq`, `is` never runs Python code.
        fn find_by_identity(&self, hash: PyHash, key: &PyObject) -> Option<usize> {
            let bucket = self.buckets.get(&hash)?;
            bucket
                .iter()
                .find(|&&idx| self.nodes[idx].key.is(key))
                .copied()
        }

        /// Every `(index, key)` sharing `hash`'s bucket, for a caller that
        /// needs to fall back to `key_eq` outside the lock.
        fn bucket_candidates(&self, hash: PyHash) -> Vec<(usize, PyObjectRef)> {
            self.buckets
                .get(&hash)
                .map(|bucket| {
                    bucket
                        .iter()
                        .map(|&idx| (idx, self.nodes[idx].key.clone()))
                        .collect()
                })
                .unwrap_or_default()
        }

        fn bucket_add(&mut self, hash: PyHash, idx: usize) {
            self.buckets.entry(hash).or_default().push(idx);
        }

        fn bucket_remove(&mut self, hash: PyHash, idx: usize) {
            if let Some(bucket) = self.buckets.get_mut(&hash) {
                if let Some(pos) = bucket.iter().position(|&i| i == idx) {
                    bucket.swap_remove(pos);
                }
                if bucket.is_empty() {
                    self.buckets.remove(&hash);
                }
            }
        }

        /// Splice node `idx` out of the recency list. Its own `prev`/`next`
        /// are left stale; every caller immediately either relinks it (`touch`,
        /// `insert`) or overwrites it (`insert` on eviction).
        fn unlink(&mut self, idx: usize) {
            let (prev, next) = (self.nodes[idx].prev, self.nodes[idx].next);
            match prev {
                Some(p) => self.nodes[p].next = next,
                None => self.front = next,
            }
            match next {
                Some(n) => self.nodes[n].prev = prev,
                None => self.back = prev,
            }
        }

        /// Link node `idx` in as the most-recently-used (back) entry.
        fn link_back(&mut self, idx: usize) {
            let old_back = self.back;
            self.nodes[idx].prev = old_back;
            self.nodes[idx].next = None;
            match old_back {
                Some(b) => self.nodes[b].next = Some(idx),
                None => self.front = Some(idx),
            }
            self.back = Some(idx);
        }

        /// Mark node `idx` most-recently-used, for a cache hit.
        fn touch(&mut self, idx: usize) {
            self.unlink(idx);
            self.link_back(idx);
        }

        /// Insert a freshly computed `(key, value)`, evicting the
        /// least-recently-used entry first if the bounded cache is full.
        /// Caller has already checked that `key` is not present.
        fn insert(
            &mut self,
            hash: PyHash,
            key: PyObjectRef,
            value: PyObjectRef,
            maxsize: LruMaxSize,
        ) {
            if let LruMaxSize::Bounded(n) = maxsize
                && self.nodes.len() >= n
            {
                let old_idx = self
                    .front
                    .expect("a full bounded lru cache always has a least-recently-used node");
                let old_hash = self.nodes[old_idx].hash;
                self.bucket_remove(old_hash, old_idx);
                self.unlink(old_idx);
                self.nodes[old_idx] = LruNode {
                    key,
                    hash,
                    value,
                    prev: None,
                    next: None,
                };
                self.bucket_add(hash, old_idx);
                self.link_back(old_idx);
                return;
            }
            let idx = self.nodes.len();
            self.nodes.push(LruNode {
                key,
                hash,
                value,
                prev: None,
                next: None,
            });
            self.bucket_add(hash, idx);
            self.link_back(idx);
        }

        fn clear(&mut self) {
            self.buckets.clear();
            self.nodes.clear();
            self.front = None;
            self.back = None;
            self.hits = 0;
            self.misses = 0;
        }
    }

    /// Builds the same cache key CPython's C `lru_cache` builds: a lone
    /// exact `str`/`int` positional argument is used bare (matching
    /// `_functoolsmodule.c`'s `lru_cache_make_key` fast path); otherwise the
    /// key is a tuple of the positional arguments, followed by
    /// `kwd_mark, name, value, ...` when there are keyword arguments, followed
    /// by each argument's `type()` when `typed` is set.
    fn make_key(
        kwd_mark: &PyObjectRef,
        args: &[PyObjectRef],
        kwargs: &KwArgs,
        typed: bool,
        vm: &VirtualMachine,
    ) -> PyObjectRef {
        let has_kwargs = !kwargs.is_empty();
        if !typed && !has_kwargs {
            if let [only] = args {
                let cls = only.class();
                if cls.is(vm.ctx.types.str_type) || cls.is(vm.ctx.types.int_type) {
                    return only.clone();
                }
            }
            return vm.ctx.new_tuple(args.to_vec()).into();
        }

        let mut key: Vec<PyObjectRef> = Vec::with_capacity(
            args.len()
                + if has_kwargs { 1 + kwargs.len() * 2 } else { 0 }
                + if typed { args.len() + kwargs.len() } else { 0 },
        );
        key.extend(args.iter().cloned());
        if has_kwargs {
            key.push(kwd_mark.clone());
            for (name, value) in kwargs {
                key.push(vm.ctx.new_str(name.clone()).into());
                key.push(value.clone());
            }
        }
        if typed {
            key.extend(args.iter().map(|a| a.class().to_owned().into()));
            if has_kwargs {
                key.extend(kwargs.values().map(|v| v.class().to_owned().into()));
            }
        }
        vm.ctx.new_tuple(key).into()
    }

    #[pyattr]
    #[pyclass(name = "_lru_cache_wrapper", module = "functools", traverse = "manual")]
    #[derive(Debug, PyPayload)]
    pub(super) struct PyLruCache {
        func: PyObjectRef,
        typed: bool,
        maxsize: LruMaxSize,
        cache_info_type: PyObjectRef,
        // Unique per-instance sentinel delimiting positional from keyword
        // arguments in a cache key, the way `_functoolsmodule.c` uses one
        // `kwd_mark` object per module (a fresh, private one per instance here
        // is just as good: it is never compared across instances).
        kwd_mark: PyObjectRef,
        state: PyMutex<LruCacheState>,
    }

    // SAFETY: every owned PyObjectRef reachable from a `PyLruCache` (the
    // wrapped function, the cache_info namedtuple type, the kwd_mark
    // sentinel, and every cached key/value) is traversed at most once.
    // (Not brought in via `use`: `object::Traverse` is implemented directly
    // for `PyRef<T>`/`PyDictRef`, and importing it makes `some_dict.clear()`
    // calls elsewhere in this module ambiguously resolve to
    // `Traverse::clear` instead of `PyDict::clear`.)
    unsafe impl crate::object::Traverse for PyLruCache {
        fn traverse(&self, tracer_fn: &mut crate::object::TraverseFn<'_>) {
            self.func.traverse(tracer_fn);
            self.cache_info_type.traverse(tracer_fn);
            self.kwd_mark.traverse(tracer_fn);
            // Best-effort: a GC traversal must never block, so a held lock
            // (e.g. from a thread paused mid-call) just skips the cached
            // entries this pass, same trade-off `PyDeque::traverse` makes.
            if let Some(state) = self.state.try_lock() {
                for node in &state.nodes {
                    node.key.traverse(tracer_fn);
                    node.value.traverse(tracer_fn);
                }
            }
        }
    }

    #[derive(FromArgs)]
    pub(super) struct LruCacheNewArgs {
        #[pyarg(any)]
        user_function: PyObjectRef,
        #[pyarg(any)]
        maxsize: PyObjectRef,
        #[pyarg(any)]
        typed: bool,
        #[pyarg(any)]
        cache_info_type: PyObjectRef,
    }

    impl Constructor for PyLruCache {
        type Args = LruCacheNewArgs;

        fn py_new(_cls: &Py<PyType>, args: Self::Args, vm: &VirtualMachine) -> PyResult<Self> {
            if !args.user_function.is_callable() {
                return Err(vm.new_type_error("the first argument must be callable"));
            }

            let maxsize = if vm.is_none(&args.maxsize) {
                LruMaxSize::Unbounded
            } else {
                let index = match args.maxsize.try_index_opt(vm) {
                    Some(result) => result?,
                    None => return Err(vm.new_type_error("maxsize should be integer or None")),
                };
                let n: isize = index.try_to_primitive(vm)?;
                match n.max(0) {
                    0 => LruMaxSize::NoCache,
                    n => LruMaxSize::Bounded(n as usize),
                }
            };

            Ok(Self {
                func: args.user_function,
                typed: args.typed,
                maxsize,
                cache_info_type: args.cache_info_type,
                kwd_mark: vm
                    .ctx
                    .new_base_object(vm.ctx.types.object_type.to_owned(), None),
                state: PyMutex::new(LruCacheState::default()),
            })
        }
    }

    #[pyclass(
        with(Constructor, Callable, GetDescriptor),
        flags(HAS_DICT, HAS_WEAKREF)
    )]
    impl PyLruCache {
        #[pygetset]
        fn __dict__(zelf: &Py<Self>, vm: &VirtualMachine) -> PyDictRef {
            zelf.as_object()
                .instance_dict()
                .map_or_else(|| vm.ctx.new_dict(), |d| d.get_or_insert(vm))
        }

        #[pygetset(setter)]
        fn set___dict__(
            zelf: &Py<Self>,
            value: PySetterValue,
            vm: &VirtualMachine,
        ) -> PyResult<()> {
            object::object_generic_set_dict(zelf.as_object().to_owned(), value, vm)
        }

        #[pymethod]
        fn cache_info(&self, vm: &VirtualMachine) -> PyResult {
            let (hits, misses, currsize) = {
                let state = self.state.lock();
                (state.hits, state.misses, state.len())
            };
            let maxsize: PyObjectRef = match self.maxsize {
                LruMaxSize::Unbounded => vm.ctx.none(),
                LruMaxSize::NoCache => vm.ctx.new_int(0).into(),
                LruMaxSize::Bounded(n) => vm.ctx.new_int(n).into(),
            };
            self.cache_info_type
                .call((hits, misses, maxsize, currsize), vm)
        }

        #[pymethod]
        fn cache_clear(&self) {
            self.state.lock().clear();
        }

        #[pymethod(name = "__reduce__")]
        fn __reduce__(zelf: &Py<Self>, vm: &VirtualMachine) -> PyResult {
            zelf.as_object().get_attr("__qualname__", vm)
        }

        #[pymethod(name = "__copy__")]
        fn __copy__(zelf: PyRef<Self>) -> PyRef<Self> {
            zelf
        }

        #[pymethod(name = "__deepcopy__")]
        fn __deepcopy__(zelf: PyRef<Self>, _memo: OptionalArg<PyObjectRef>) -> PyRef<Self> {
            zelf
        }
    }

    impl GetDescriptor for PyLruCache {
        fn descr_get(
            zelf: PyObjectRef,
            obj: Option<PyObjectRef>,
            _cls: Option<PyObjectRef>,
            vm: &VirtualMachine,
        ) -> PyResult {
            let obj = match obj {
                Some(obj) if !vm.is_none(&obj) => obj,
                _ => return Ok(zelf),
            };
            Ok(PyBoundMethod::new(obj, zelf).into_ref(&vm.ctx).into())
        }
    }

    impl Callable for PyLruCache {
        type Args = FuncArgs;

        fn call(zelf: &Py<Self>, args: FuncArgs, vm: &VirtualMachine) -> PyResult {
            if zelf.maxsize == LruMaxSize::NoCache {
                zelf.state.lock().misses += 1;
                return zelf.func.call(args, vm);
            }

            let key = make_key(&zelf.kwd_mark, &args.args, &args.kwargs, zelf.typed, vm);
            // Computed once and reused for every subsequent lookup this call,
            // matching CPython's guarantee that `__hash__` runs at most once
            // per `lru_cache`-wrapped call.
            let hash = key.key_hash(vm)?;

            if let Some(value) = zelf.lookup_and_touch(hash, &key, vm)? {
                return Ok(value);
            }

            // Released the lock before running arbitrary Python code: the
            // wrapped function may legitimately recurse back into this same
            // cache (see CPython's lru_cache bpo-35780 regression test).
            let result = zelf.func.call(args, vm)?;

            zelf.insert_if_absent(hash, key, result.clone(), vm)?;

            Ok(result)
        }
    }

    impl PyLruCache {
        /// Look up `key` (with precomputed `hash`). On a hit, marks the entry
        /// most-recently-used, counts it, and returns its value; on a miss,
        /// counts it and returns `None`.
        ///
        /// `key_eq` can run arbitrary Python code -- including a call that
        /// legitimately reenters this very cache on the same thread, the
        /// scenario CPython's lru_cache `test_need_for_rlock` exists to guard
        /// against -- so it only ever runs with `self.state` unlocked. The
        /// cheap identity check (`is`) that never runs Python code is the
        /// only comparison made while locked.
        fn lookup_and_touch(
            &self,
            hash: PyHash,
            key: &PyObject,
            vm: &VirtualMachine,
        ) -> PyResult<Option<PyObjectRef>> {
            let candidates = {
                let mut state = self.state.lock();
                if let Some(idx) = state.find_by_identity(hash, key) {
                    let value = state.nodes[idx].value.clone();
                    state.touch(idx);
                    state.hits += 1;
                    return Ok(Some(value));
                }
                let candidates = state.bucket_candidates(hash);
                if candidates.is_empty() {
                    state.misses += 1;
                    return Ok(None);
                }
                candidates
            };

            let mut matched = None;
            for (idx, candidate_key) in candidates {
                if key.key_eq(vm, &candidate_key)? {
                    matched = Some((idx, candidate_key));
                    break;
                }
            }

            let mut state = self.state.lock();
            // Re-validate under the lock: eviction (from a concurrent or
            // reentrant call) may have reused this exact slot for a
            // different key while `key_eq` ran unlocked above.
            if let Some((idx, candidate_key)) = matched
                && state
                    .nodes
                    .get(idx)
                    .is_some_and(|node| node.key.is(&candidate_key))
            {
                let value = state.nodes[idx].value.clone();
                state.touch(idx);
                state.hits += 1;
                return Ok(Some(value));
            }
            state.misses += 1;
            Ok(None)
        }

        /// Cache a freshly computed `(key, value)`, unless `key` is already
        /// present -- meaning a concurrent or reentrant call cached it while
        /// the wrapped function ran -- in which case the cache is left
        /// untouched, matching CPython's `bounded_lru_cache_update_lock_held`
        /// (bpo-35780). Same locked-identity/unlocked-`key_eq` split as
        /// `lookup_and_touch`, for the same reentrancy reason.
        fn insert_if_absent(
            &self,
            hash: PyHash,
            key: PyObjectRef,
            value: PyObjectRef,
            vm: &VirtualMachine,
        ) -> PyResult<()> {
            let candidates = {
                let mut state = self.state.lock();
                if state.find_by_identity(hash, &key).is_some() {
                    return Ok(());
                }
                let candidates = state.bucket_candidates(hash);
                if candidates.is_empty() {
                    state.insert(hash, key, value, self.maxsize);
                    return Ok(());
                }
                candidates
            };

            let mut already_present = false;
            for (_, candidate_key) in candidates {
                if key.key_eq(vm, &candidate_key)? {
                    already_present = true;
                    break;
                }
            }

            let mut state = self.state.lock();
            if !already_present && state.find_by_identity(hash, &key).is_none() {
                state.insert(hash, key, value, self.maxsize);
            }
            Ok(())
        }
    }
}
