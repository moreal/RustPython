use rustpython_vm::{
    AsObject, Context, Interpreter, eval,
    object::{debug_assertion, debug_init_type_hierarchy},
    types::TypeZoo,
};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eval(s: *const u8, l: usize) -> u32 {
    let src = std::slice::from_raw_parts(s, l);
    let Ok(src) = std::str::from_utf8(src) else {
        return 0;
    };
    // return debug_assertion();
    // let zoo = debug_init_type_hierarchy();
    // let context = Context::genesis();
    let interpreter = Interpreter::without_stdlib(Default::default());
    return interpreter.enter(|vm| {
        let res = match eval::eval(vm, src, vm.new_scope_with_builtins(), "<string>") {
            Ok(res) => res,
            Err(e) => {
                if e.class().is(vm.ctx.exceptions.syntax_error) {
                    return 4001;
                } else {
                    return 4000;
                }
            }
        };
        res.try_into_value(vm).unwrap_or(3000)
    });
}

#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(
    _dest: *mut u8,
    _len: usize,
) -> Result<(), getrandom::Error> {
    return Ok(());
    // Err(getrandom::Error::UNSUPPORTED)
}
