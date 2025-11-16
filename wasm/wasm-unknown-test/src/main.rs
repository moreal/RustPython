use rustpython_vm::{AsObject, Context, Interpreter, eval, types::TypeZoo};

fn main() -> () {
    let src = "1";
    let interpreter = Interpreter::without_stdlib(Default::default());
    let return_value: u32 = interpreter.enter(|vm| {
        let res = match eval::eval(vm, src, vm.new_scope_with_builtins(), "<string>") {
            Ok(res) => res,
            Err(e) => {
                println!("error");
                if e.class().is(vm.ctx.exceptions.syntax_error) {
                    println!("Syntax error: {:?}", e);
                    return 40001;
                } else {
                    return 4000;
                }
            }
        };
        res.try_into_value(vm).unwrap_or(3000)
    });
    println!("Return Value: {:?}", return_value);
}
