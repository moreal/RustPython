//! The compiler's constant cache (`c_const_cache`).
//!
//! Equal constants and code fields produced by one compilation share a single
//! object, merged under their `_PyCode_ConstantKey`.

use super::*;
use crate::builtins::{PyDict, PyDictRef, PyTuple};
use crate::dict_inner::DictKey;

/// A lookup key used only after bottom-up compilation has canonicalized every
/// child constant and each code object's co_consts tuple. Equal code constants
/// then share that tuple's identity, so hashing need not revisit descendants.
/// The dictionary still stores the original constant key and uses its equality.
/// A tuple input is the two-element key made by constant_key for an exact tuple.
struct CompilerKey<'a>(&'a PyObject);

fn compiler_hash(op: &PyObject, vm: &VirtualMachine) -> PyResult<crate::common::hash::PyHash> {
    if let Some(tuple) = op.downcast_ref_if_exact::<PyTuple>(vm) {
        crate::common::hash::hash_tuple(tuple.iter().map(|item| compiler_hash(item, vm)))
    } else if let Some(code) = op.downcast_ref::<PyCode>() {
        hash_code_with_constants(code, vm, || {
            Ok(crate::common::hash::hash_object_id(
                code.co_consts().get_id(),
            ))
        })
    } else {
        op.hash(vm)
    }
}

impl DictKey for CompilerKey<'_> {
    type Owned = PyObjectRef;

    fn _to_owned(&self, vm: &VirtualMachine) -> Self::Owned {
        self.0._to_owned(vm)
    }

    fn key_hash(&self, vm: &VirtualMachine) -> PyResult<crate::common::hash::PyHash> {
        let key = match self.0.downcast_ref_if_exact::<PyTuple>(vm) {
            // Equal tuple keys have equal element keys.
            // Keep the original tuple for equality and decoding, not hashing.
            Some(key) => &key.as_slice()[0],
            None => self.0,
        };
        compiler_hash(key, vm)
    }

    fn key_is(&self, other: &PyObject) -> bool {
        self.0.key_is(other)
    }

    fn key_eq(&self, vm: &VirtualMachine, other: &PyObject) -> PyResult<bool> {
        self.0.key_eq(vm, other)
    }

    fn key_as_isize(&self, vm: &VirtualMachine) -> PyResult<isize> {
        self.0.key_as_isize(vm)
    }
}

fn constant_from_key(key: PyObjectRef, vm: &VirtualMachine) -> PyObjectRef {
    // See constant_key().
    match key.downcast_ref_if_exact::<PyTuple>(vm) {
        Some(key) => key.as_slice()[1].clone(),
        None => key,
    }
}

/// `const_cache_insert`. Items of a tuple or frozenset constant are created
/// through the cache before the container, so a single non-recursive insert
/// also merges them.
fn const_cache_insert(
    const_cache: &Py<PyDict>,
    o: PyObjectRef,
    vm: &VirtualMachine,
) -> PyResult<PyObjectRef> {
    // None and Ellipsis are singletons, and key is the singleton.
    if vm.is_none(&o) || o.is(&vm.ctx.ellipsis) {
        return Ok(o);
    }
    let key = constant_key(&o, vm)?;
    let key = const_cache.get_or_insert(vm, &key, || key.clone())?;
    Ok(constant_from_key(key, vm))
}

/// A [`ConstantBag`] creating the constants of one compilation through its
/// constant cache.
#[derive(Clone, Copy)]
struct CompileBag<'a> {
    vm: &'a VirtualMachine,
    const_cache: &'a Py<PyDict>,
}

impl CompileBag<'_> {
    /// `_PyCompile_ConstCacheMergeOne`
    fn merge_one<T: PyPayload>(&self, obj: PyRef<T>) -> PyRef<T> {
        self.merge(obj.into())
            .downcast()
            .expect("an equal constant key implies the same type")
    }

    fn merge(&self, obj: PyObjectRef) -> PyObjectRef {
        let result = if obj.class().is(self.vm.ctx.types.tuple_type)
            || obj.class().is(self.vm.ctx.types.code_type)
        {
            self.merge_compiler_key(obj)
        } else {
            // Scalar and opaque container keys use their usual hash. It is
            // also what compiler_hash uses when it encounters them in tuples.
            const_cache_insert(self.const_cache, obj, self.vm)
        };
        result.expect("constant keys are hashable and comparable")
    }

    fn merge_compiler_key(&self, obj: PyObjectRef) -> PyResult<PyObjectRef> {
        let key = constant_key(&obj, self.vm)?;
        let lookup = CompilerKey(&key);
        let hash = lookup.key_hash(self.vm)?;
        let key = self
            .const_cache
            .entries
            .setdefault(self.vm, &lookup, hash, || key.clone())?;
        Ok(constant_from_key(key, self.vm))
    }

    /// Create a code object whose fields are merged the way `assemble.c`
    /// merges them, in the same order.
    fn new_code(&self, code: CodeObject) -> PyCode {
        let mut code = PyCode::new(code, &self.vm.ctx);
        code.apply_debug_ranges(self.vm);
        code.co_exceptiontable = self.merge_one(code.co_exceptiontable);
        code.co_linetable = self.merge_one(code.co_linetable);
        code.co_names = self.merge_one(code.co_names);
        let consts = self.merge_one(code.code.constants.as_tuple().as_pytuple().clone());
        code.code.constants = bytecode::Constants::new(LiteralTuple(consts));
        code.co_localsplusnames = self.merge_one(code.co_localsplusnames);
        code
    }
}

impl ConstantBag for CompileBag<'_> {
    type Constant = Literal;

    fn make_constant<C: Constant>(&self, constant: BorrowedConstant<'_, C>) -> Literal {
        let vm = self.vm;
        let obj = match constant {
            BorrowedConstant::Code { code } => return self.make_code(code.map_clone_bag(self)),
            BorrowedConstant::Tuple { elements } => vm
                .ctx
                .new_tuple(
                    elements
                        .iter()
                        .map(|item| self.make_constant(item.borrow_constant()).0)
                        .collect(),
                )
                .into(),
            BorrowedConstant::Frozenset { elements } => PyFrozenSet::from_iter(
                vm,
                elements
                    .iter()
                    .map(|item| self.make_constant(item.borrow_constant()).0),
            )
            .unwrap()
            .into_ref(&vm.ctx)
            .into(),
            BorrowedConstant::Slice { elements } => {
                let [start, stop, step] = elements
                    .each_ref()
                    .map(|item| self.make_constant(item.borrow_constant()).0);
                crate::builtins::PySlice {
                    start: Some(start),
                    stop,
                    step: Some(step),
                }
                .into_ref(&vm.ctx)
                .into()
            }
            _ => PyVmBag(vm).make_constant(constant).0,
        };
        Literal(self.merge(obj))
    }

    fn make_name(&self, name: &str) -> &'static PyStrInterned {
        self.vm.ctx.intern_str(name)
    }

    fn make_int(&self, value: BigInt) -> Literal {
        Literal(self.merge(self.vm.ctx.new_int(value).into()))
    }

    fn make_tuple(&self, elements: impl Iterator<Item = Literal>) -> Literal {
        let tuple = self.vm.ctx.new_tuple(elements.map(|item| item.0).collect());
        Literal(self.merge(tuple.into()))
    }

    fn make_code(&self, code: CodeObject) -> Literal {
        let code = self.new_code(code).into_ref(&self.vm.ctx);
        Literal(self.merge(code.into()))
    }
}

pub(super) fn into_code_object(code: bytecode::CodeObject, vm: &VirtualMachine) -> CodeObject {
    let const_cache: PyDictRef = vm.ctx.new_dict();
    code.map_bag(CompileBag {
        vm,
        const_cache: &const_cache,
    })
}

pub(super) fn into_pycode(code: bytecode::CodeObject, vm: &VirtualMachine) -> PyCode {
    let const_cache: PyDictRef = vm.ctx.new_dict();
    let bag = CompileBag {
        vm,
        const_cache: &const_cache,
    };
    bag.new_code(code.map_bag(bag))
}

#[cfg(test)]
#[cfg(feature = "compiler")]
mod tests {
    use super::*;
    use crate::{Interpreter, builtins::PyStr, scope::Scope};

    fn run(vm: &VirtualMachine, scope: &Scope, source: &str) {
        let code = vm
            .compile(source, crate::compiler::Mode::Exec, "<test>")
            .map_err(|error| error.into_pyexception(vm, Some(source)))
            .unwrap();
        vm.unwrap_pyresult(vm.run_code_obj(code, scope.clone()));
    }

    const SOURCE: &str = r#"
failure = ValueError("constant hash failed")

class HashCounter:
    calls = 0
    fail = False

    def __hash__(self):
        self.calls += 1
        if self.fail:
            raise failure
        return 17

counter = HashCounter()
code = (lambda: None).__code__.replace(co_consts=(counter,))
equal_code = code.replace()
assert code is not equal_code
assert code == equal_code
"#;

    #[test]
    fn compiler_cache_hashes_code_chains_without_revisiting_descendants() {
        Interpreter::without_stdlib(Default::default()).enter(|vm| {
            let scope = vm.new_scope_with_builtins();
            run(
                vm,
                &scope,
                r#"
class HashName(str):
    calls = 0

    def __hash__(self):
        type(self).calls += 1
        return 17

name = HashName("function")
"#,
            );
            let name = scope
                .globals
                .get_item("name", vm)
                .unwrap()
                .downcast::<PyStr>()
                .unwrap();
            let base = vm
                .compile("pass", crate::compiler::Mode::Exec, "<test>")
                .unwrap();
            let cache = vm.ctx.new_dict();
            let bag = CompileBag {
                vm,
                const_cache: &cache,
            };
            let mut child = vm.ctx.none();
            for _ in 0..12 {
                let mut code = base.code.clone();
                code.constants = bytecode::Constants::new(LiteralTuple(
                    vm.ctx.new_tuple(vec![vm.ctx.none(), child]),
                ));
                let mut code = bag.new_code(code);
                code.co_name = name.clone();
                child = bag.merge(code.into_ref(&vm.ctx).into());
            }
            run(vm, &scope, "assert HashName.calls <= 36, HashName.calls");
        });
    }

    #[test]
    fn compiler_tuple_lookup_hashes_element_keys_only_once() {
        Interpreter::without_stdlib(Default::default()).enter(|vm| {
            let scope = vm.new_scope_with_builtins();
            run(
                vm,
                &scope,
                r#"
class HashName(str):
    calls = 0

    def __hash__(self):
        type(self).calls += 1
        return 17

name = HashName("function")
"#,
            );
            let name = scope
                .globals
                .get_item("name", vm)
                .unwrap()
                .downcast::<PyStr>()
                .unwrap();
            let base = vm
                .compile("pass", crate::compiler::Mode::Exec, "<test>")
                .unwrap();
            let cache = vm.ctx.new_dict();
            let bag = CompileBag {
                vm,
                const_cache: &cache,
            };
            let mut code = bag.new_code(base.code.clone());
            code.co_name = name;
            let child = Literal(bag.merge(code.into_ref(&vm.ctx).into()));
            run(vm, &scope, "HashName.calls = 0");
            bag.make_tuple(core::iter::once(child));
            run(vm, &scope, "assert HashName.calls == 1, HashName.calls");
        });
    }

    #[test]
    fn compiler_cache_merges_equal_nested_code_but_keeps_distinct_values() {
        Interpreter::without_stdlib(Default::default()).enter(|vm| {
            let base = vm
                .compile("pass", crate::compiler::Mode::Exec, "<test>")
                .unwrap();
            let cache = vm.ctx.new_dict();
            let bag = CompileBag {
                vm,
                const_cache: &cache,
            };
            let make_leaf = |value| {
                let mut code = base.code.clone();
                code.constants = bytecode::Constants::new(LiteralTuple(
                    vm.ctx.new_tuple(vec![vm.ctx.new_int(value).into()]),
                ));
                bag.make_code(code).0
            };
            let first = make_leaf(12345);
            let duplicate = make_leaf(12345);
            let different = make_leaf(67890);
            assert!(first.is(&duplicate));
            assert!(!first.is(&different));
            let make_parent = |child| {
                let mut code = base.code.clone();
                code.constants =
                    bytecode::Constants::new(LiteralTuple(vm.ctx.new_tuple(vec![child])));
                bag.make_code(code).0
            };
            let parent = make_parent(first);
            assert!(parent.is(&make_parent(duplicate)));
            assert!(!parent.is(&make_parent(different)));
        });
    }

    #[test]
    fn compiler_cache_preserves_numeric_keys_inside_code_and_containers() {
        use bytecode::ConstantData;

        Interpreter::without_stdlib(Default::default()).enter(|vm| {
            let base = crate::compiler::compile(
                "pass",
                crate::compiler::Mode::Exec,
                "<test>",
                Default::default(),
            )
            .unwrap();
            let cache = vm.ctx.new_dict();
            let bag = CompileBag {
                vm,
                const_cache: &cache,
            };
            let wrap = |value, container| {
                let mut code = base.clone();
                code.constants = core::iter::once(value).collect();
                let code = ConstantData::Code {
                    code: Box::new(code),
                };
                match container {
                    0 => code,
                    1 => ConstantData::Tuple {
                        elements: vec![code],
                    },
                    2 => ConstantData::Frozenset {
                        elements: vec![code],
                    },
                    3 => ConstantData::Slice {
                        elements: Box::new([code, ConstantData::None, ConstantData::None]),
                    },
                    _ => unreachable!(),
                }
            };
            let mut values = vec![
                ConstantData::Boolean { value: true },
                ConstantData::Integer { value: 1.into() },
                ConstantData::Float { value: 0.0 },
                ConstantData::Float { value: -0.0 },
            ];
            for re in [0.0, -0.0] {
                for im in [0.0, -0.0] {
                    values.push(ConstantData::Complex {
                        value: num_complex::Complex64::new(re, im),
                    });
                }
            }
            for container in 0..4 {
                let mut previous: Vec<PyObjectRef> = Vec::new();
                for value in &values {
                    let source = wrap(value.clone(), container);
                    let first = bag.make_constant(source.borrow_constant()).0;
                    let duplicate = bag.make_constant(source.borrow_constant()).0;
                    assert!(first.is(&duplicate));
                    assert!(previous.iter().all(|other| !first.is(other)));
                    previous.push(first);
                }
                let nan = wrap(ConstantData::Float { value: f64::NAN }, container);
                let first = bag.make_constant(nan.borrow_constant()).0;
                let second = bag.make_constant(nan.borrow_constant()).0;
                assert!(!first.is(&second));
            }
        });
    }

    #[test]
    fn code_comparison_preserves_hash_errors_with_shared_constants() {
        Interpreter::without_stdlib(Default::default()).enter(|vm| {
            let scope = vm.new_scope_with_builtins();
            run(
                vm,
                &scope,
                r#"
failure = ValueError("name hash failed")

class HashName(str):
    fail = False

    def __hash__(self):
        if type(self).fail:
            raise failure
        return 17

base = (lambda: None).__code__
inner = base.replace(co_name=HashName("inner"))
outer = base.replace(co_consts=(frozenset((inner,)),))
other = outer.replace()
assert outer is not other
assert outer.co_consts is other.co_consts
assert outer == other
HashName.fail = True
for compare in (lambda: outer == other, lambda: outer != other):
    try:
        compare()
    except ValueError as error:
        assert error is failure
    else:
        raise AssertionError("comparison must rebuild the frozenset constant key")
"#,
            );
        });
    }

    #[test]
    fn insertion_hashes_once_and_returns_existing_constant() {
        Interpreter::without_stdlib(Default::default()).enter(|vm| {
            let scope = vm.new_scope_with_builtins();
            run(vm, &scope, SOURCE);
            let cache = vm.ctx.new_dict();
            let code = scope.globals.get_item("code", vm).unwrap();
            let equal_code = scope.globals.get_item("equal_code", vm).unwrap();

            let inserted = const_cache_insert(&cache, code.clone(), vm).unwrap();
            assert!(inserted.is(&code));
            run(vm, &scope, "assert counter.calls == 1, counter.calls");

            let existing = const_cache_insert(&cache, equal_code, vm).unwrap();
            assert!(existing.is(&code));
            run(vm, &scope, "assert counter.calls == 2, counter.calls");
        });
    }

    #[test]
    fn insertion_propagates_hash_errors_on_misses_and_hits() {
        Interpreter::without_stdlib(Default::default()).enter(|vm| {
            let scope = vm.new_scope_with_builtins();
            run(vm, &scope, SOURCE);
            let cache = vm.ctx.new_dict();
            let code = scope.globals.get_item("code", vm).unwrap();
            let failure = scope.globals.get_item("failure", vm).unwrap();

            run(vm, &scope, "counter.fail = True");
            let error = const_cache_insert(&cache, code.clone(), vm).unwrap_err();
            assert!(error.as_object().is(&failure));
            assert!(cache.is_empty());
            run(vm, &scope, "assert counter.calls == 1, counter.calls");

            run(vm, &scope, "counter.fail = False");
            const_cache_insert(&cache, code.clone(), vm).unwrap();
            run(vm, &scope, "assert counter.calls == 2, counter.calls");

            run(vm, &scope, "counter.fail = True");
            let error = const_cache_insert(&cache, code, vm).unwrap_err();
            assert!(error.as_object().is(&failure));
            assert!(!cache.is_empty());
            run(vm, &scope, "assert counter.calls == 3, counter.calls");
        });
    }

    #[test]
    fn code_hash_uses_constants_slice_without_tuple_subclass_callbacks() {
        Interpreter::without_stdlib(Default::default()).enter(|vm| {
            let scope = vm.new_scope_with_builtins();
            run(
                vm,
                &scope,
                r#"
class Constants(tuple):
    def __hash__(self):
        raise AssertionError("constants tuple hash must not be called")

    def __iter__(self):
        raise AssertionError("constants tuple iterator must not be called")

def function(a, /, *args, b=1, **kwargs):
    return a + b

values = (None, 1, 2)
code = function.__code__.replace(co_consts=Constants(values))
assert type(code.co_consts) is Constants
expected = hash((code.co_name, code.co_argcount, code.co_posonlyargcount,
                 code.co_kwonlyargcount, code.co_nlocals, code.co_flags,
                 code.co_firstlineno, code.co_code, values))
assert hash(code) == expected
"#,
            );
        });
    }

    #[test]
    fn code_hash_short_circuits_custom_callbacks_in_attribute_order() {
        Interpreter::without_stdlib(Default::default()).enter(|vm| {
            let scope = vm.new_scope_with_builtins();
            run(
                vm,
                &scope,
                r#"
events = []
failure = ValueError("hash failed")

class Name(str):
    fail = False

    def __hash__(self):
        events.append("name")
        if self.fail:
            raise failure
        return 11

class Value:
    fail = False

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        events.append(self.name)
        if self.fail:
            raise failure
        return 17

name = Name("function")
first, second = Value("first"), Value("second")
code = (lambda: None).__code__.replace(co_name=name, co_consts=(first, second))
hash(code)
assert events == ["name", "first", "second"], events

events.clear()
first.fail = True
try:
    hash(code)
except ValueError as error:
    assert error is failure
else:
    raise AssertionError("hash must propagate the constant failure")
assert events == ["name", "first"], events

events.clear()
name.fail = True
try:
    hash(code)
except ValueError as error:
    assert error is failure
else:
    raise AssertionError("hash must propagate the name failure")
assert events == ["name"], events
"#,
            );
        });
    }

    #[test]
    fn code_hash_recomputes_custom_constant_hashes() {
        Interpreter::without_stdlib(Default::default()).enter(|vm| {
            let scope = vm.new_scope_with_builtins();
            run(vm, &scope, SOURCE);
            run(
                vm,
                &scope,
                r#"
hash(code)
hash(code)
assert counter.calls == 2, counter.calls
counter.fail = True
try:
    hash(code)
except ValueError as error:
    assert error is failure
else:
    raise AssertionError("code hash must not cache a custom constant")
assert counter.calls == 3, counter.calls
"#,
            );
        });
    }
}
