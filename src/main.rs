extern crate lalrpop_util;

mod ast;
mod noise_parser;
pub mod noise;
mod engine;


fn main() {
    let mut engine = noise::Engine::new();
    test1(&mut engine);
    test2(&mut engine);
    test3(&mut engine);
}

fn test1(engine: &mut noise::Engine) {
    let test = "A=10 B=10 A+B";
    let res = engine.expr(test).unwrap();
    println!("TEST1: {} => {:?}", test, res);
}

fn test2(engine: &mut noise::Engine) {
    let test = "24*(1+1) + (5+5)";
    let res = engine.expr(test).unwrap();
    println!("TEST2: {} => {:?}", test, res);
}

fn test3(engine: &mut noise::Engine) {
    let test = "a=1 b=2 c=b*a c+a";
    let res = engine.expr(test).unwrap();
    println!("TEST3: {} => {:?}", test, res);
}

