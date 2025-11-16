use rustpython_vm::{AsObject, Context, Interpreter, convert::IntoObject, eval, types::TypeZoo};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn eval(s: *const u8, l: usize) -> u32 {
    let src = std::slice::from_raw_parts(s, l);
    let Ok(src) = std::str::from_utf8(src) else {
        return 0;
    };
    let interpreter = Interpreter::without_stdlib(Default::default());
    return interpreter.enter(|vm| {
        let res = eval::eval(vm, "2+10000", vm.new_scope_with_builtins(), "<string>").unwrap();
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
