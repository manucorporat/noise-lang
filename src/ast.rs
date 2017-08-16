use std::fmt::{Debug, Formatter, Error};

pub struct Unit {
    pub exprs: Vec<Box<Expr>>,
}

impl Unit {
    pub fn new(exprs: Vec<Box<Expr>>) -> Unit {
        Unit{exprs: exprs}
    }
}

pub enum Expr {
  Number(f64),
  Op(Box<Expr>, Opcode, Box<Expr>),
  Id(String),
  Error,
}

#[derive(Copy, Clone)]
pub enum Opcode {
  Assign,
  Ref,

  Mul,
  Div,
  Add,
  Sub,
}

impl Debug for Expr {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Expr::*;
        match self {
            &Number(n) => write!(fmt, "{:?}", n),
            &Id(ref id) => write!(fmt, "{:?}", id),
            &Op(ref l, op, ref r) => write!(fmt, "({:?} {:?} {:?})", l, op, r),
            &Error => write!(fmt, "error"),
        }
    }
}

impl Debug for Opcode {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        use self::Opcode::*;
        match *self {
          Assign => write!(fmt, "="),
            Ref => write!(fmt, "~"),

            Mul => write!(fmt, "*"),
            Div => write!(fmt, "/"),
            Add => write!(fmt, "+"),
            Sub => write!(fmt, "-"),
        }
    }
}
