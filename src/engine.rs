
use ast::{Expr, Opcode};
use noise_parser;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Ret {
    Number(f64),
    Text(String),
    RandomVar,
}

pub struct Engine {
    identifiers: HashMap<String, Box<Ret>>
}


impl Engine {

  pub fn new() -> Engine {
    return Engine{
      identifiers: HashMap::new()
    }
  }

  pub fn parse(input: &str) -> Expr {
    *noise_parser::parse_Unit(input).unwrap()
  }

  pub fn expr(&mut self, input: &str) -> Ret {
    let ast = Engine::parse(input);
    self.run_unit(&ast)
  }

  pub fn run_unit(&mut self, expr: &Expr) -> Ret {
    self.compute(expr)
  }

  fn compute(&mut self, node: &Expr) -> Ret {
    use self::Expr::*;
    match node {
        &Number(x) => Ret::Number(x),
        &Text(ref text) => Ret::Number(0.0),
        &Id(ref id) => self.resolve_id(id),
        &List(ref list) => self.op_list(list),
        &Assign(ref id, op, ref e) => self.op_assign(id, e.as_ref()),
        &Op(ref e1, op, ref e2) => self.op_bi(e1.as_ref(), op, e2.as_ref()),
        &Error => panic!("ERROR")
    }
  }

  fn op_list(&mut self, list: &Vec<Box<Expr>>) -> Ret {
    let mut last_ret = Ret::Number(0.0);
    for expr in list.iter() {
      last_ret = self.compute(&expr);
    }
    last_ret
  }

  fn resolve_id(&self, id: &String) -> Ret {
    (*self.identifiers.get(id).unwrap().as_ref()).clone()
  }

  fn op_bi(&mut self, e1: &Expr, op: Opcode, e2: &Expr) -> Ret {
    let left = self.compute(e1);
    let right = self.compute(e2);
    op_compute(&left, op, &right)
  }

  fn op_assign(&mut self, id: &String, e: &Expr) -> Ret {
    let right = self.compute(e);
    self.identifiers.insert(id.clone(), Box::new(right.clone()));
    println!("ASSIGN: {} {:?}", id, &right);
    right.clone()
  }
}


fn op_compute(e1: &Ret, op: Opcode, e2: &Ret) -> Ret {
  use self::Opcode::*;
  match op {
    Add => op_sum(e1, e2),
    Sub => op_sub(e1, e2),
    Mul => op_mult(e1, e2),
    Div => op_div(e1, e2),
    _ => panic!("hola")
  }
}


fn op_sum(e1: &Ret, e2: &Ret) -> Ret {
  use self::Ret::*;
  match (e1, e2) {
    (&Number(x1), &Number(x2)) => Number(x1+x2),

    _ => panic!("ERROR")
  }
}

fn op_sub(e1: &Ret, e2: &Ret) -> Ret {
  use self::Ret::*;
  match (e1, e2) {
    (&Number(x1), &Number(x2)) => Number(x1-x2),
    _ => panic!("ERROR")
  }
}

fn op_mult(e1: &Ret, e2: &Ret) -> Ret {
  use self::Ret::*;
  match (e1, e2) {
    (&Number(x1), &Number(x2)) => Number(x1*x2),
    _ => panic!("ERROR")
  }
}

fn op_div(e1: &Ret, e2: &Ret) -> Ret {
  use self::Ret::*;
  match (e1, e2) {
    (&Number(x1), &Number(x2)) => Number(x1/x2),
    _ => panic!("ERROR")
  }
}
