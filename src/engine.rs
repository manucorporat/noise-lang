
use ast::{Expr, Opcode, Unit};
use noise_parser;
use std::collections::HashMap;

#[derive(Debug, Clone)]
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

  pub fn parse(input: &str) -> Unit {
    noise_parser::parse_Unit(input).unwrap()
  }

  pub fn expr(&mut self, input: &str) -> Option<Ret> {
    let ast = Engine::parse(input);
    self.run_unit(&ast)
  }

  pub fn run_unit(&mut self, unit: &Unit) -> Option<Ret> {
    let mut last_ret = Ret::Number(0.0);
    for expr in unit.exprs.iter() {
      last_ret = self.compute(&expr);
    }
    Some(last_ret)
  }

  fn compute(&mut self, node: &Expr) -> Ret {
    use self::Expr::*;
    match node {
        &Number(x) => Ret::Number(x),
        &Id(ref id) => self.resolve_id(id),
        &Op(ref e1, op, ref e2) => self.op_bi(e1.as_ref(), op, e2.as_ref()),
        &Error => panic!("ERROR")
    }
  }

  fn op_bi(&mut self, e1: &Expr, op: Opcode, e2: &Expr) -> Ret {
    use self::Opcode::*;
    use self::Expr::*;

    let right = self.compute(e2);
    match (op, e1) {
      (Assign, &Id(ref id)) => self.op_assign(id, &right),
      _ => {
        let left = self.compute(e1);
        self.op_compute(&left, op, &right)
        },
      }
  }

  fn resolve_id(&self, id: &String) -> Ret {
    (*self.identifiers.get(id).unwrap().as_ref()).clone()
  }

  fn op_compute(&self, e1: &Ret, op: Opcode, e2: &Ret) -> Ret {
    use self::Opcode::*;
    match op {
      Add => op_sum(e1, e2),
      Sub => op_sub(e1, e2),
      Mul => op_mult(e1, e2),
      Div => op_div(e1, e2),
      _ => panic!("hola")
    }
  }

  fn op_assign(&mut self, id: &String, e: &Ret) -> Ret {
    self.identifiers.insert(id.clone(), Box::new(e.clone()));
    e.clone()
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
