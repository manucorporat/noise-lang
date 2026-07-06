mod ast;
mod noise_parser;
mod noise;
mod engine;

mod tests {
    use super::noise::*;
    use super::noise::Ret::*;

    fn check(expr: &str, ret: Ret) {
        let mut e = Engine::new();
        assert_eq!(e.expr(expr), ret);
    }

    #[test]
    fn arith() {
        let mut e = Engine::new();
        assert_eq!(e.expr("(3*(2+1)+9)/2;"), Number(9.0));
        assert_eq!(e.expr("3*(2+1)+9/2;"), Number(13.5));
        assert_eq!(e.expr("3*2+1+9/2;"), Number(11.5));
        assert_eq!(e.expr("3*2+1-9/2-4;"), Number(-1.5));
    }

    #[test]
    fn assigment() {
        check("a=1+(2*3);a;", Number(7.0));
        check("a=2;a;", Number(2.0));
        check("a=2;", Number(2.0));
    }

    #[test]
    fn assigment_multiple() {
        check("a=1+(2*3);b=a+2;a+b;", Number(16.0));
    }

    #[test]
    fn block() {
        check("{1;2;3;};", Number(3.0));
        check("a={1;2;3;};a;", Number(3.0));
        check("a={b=2*3;c=(b+1)/2;}; b+c;};", Number(9.5));
        check("a={b=2*3;c=(b+1)/2;}; b+c;};", Number(9.5));
    }


}
