use std::str::FromStr;
use ast::{Expr, Opcode};
extern crate lalrpop_util as __lalrpop_util;

mod __parse__Unit {
    #![allow(non_snake_case, non_camel_case_types, unused_mut, unused_variables, unused_imports)]

    use std::str::FromStr;
    use ast::{Expr, Opcode};
    extern crate lalrpop_util as __lalrpop_util;
    #[allow(dead_code)]
    pub enum __Symbol<'input> {
        Term_22_28_22(&'input str),
        Term_22_29_22(&'input str),
        Term_22_2a_22(&'input str),
        Term_22_2b_22(&'input str),
        Term_22_2d_22(&'input str),
        Term_22_2f_22(&'input str),
        Term_22_3b_22(&'input str),
        Term_22_3d_22(&'input str),
        Term_22_7b_22(&'input str),
        Term_22_7d_22(&'input str),
        Term_22_7e_22(&'input str),
        Termr_23_22_5b0_2d9_5d_2b_22_23(&'input str),
        Termr_23_22_5ba_2dzA_2dZ_5d_5ba_2dzA_2dZ0_2d9_5d_2a_22_23(&'input str),
        Termr_23_22_5c_5cx22_5b_5e_5c_5cx22_5d_2a_5c_5cx22_22_23(&'input str),
        NtAssign(Box<Expr>),
        NtAssignOp(Opcode),
        NtBlock(Box<Expr>),
        NtExpr(Box<Expr>),
        NtExprOp(Opcode),
        NtFactor(Box<Expr>),
        NtFactorOp(Opcode),
        NtFullStatement(Box<Expr>),
        NtFullStatement_2a(::std::vec::Vec<Box<Expr>>),
        NtFullStatement_2b(::std::vec::Vec<Box<Expr>>),
        NtIdentifier(String),
        NtLiteral(Box<Expr>),
        NtMathExpr(Box<Expr>),
        NtNum(Box<Expr>),
        NtStatement(Box<Expr>),
        NtStatements(Box<Expr>),
        NtSubExpr(Box<Expr>),
        NtText(Box<Expr>),
        NtUnit(Box<Expr>),
        Nt____Unit(Box<Expr>),
    }
    const __ACTION: &'static [i32] = &[
        // State 0
        17, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 19, 20, 21,
        // State 1
        -26, -26, -26, -26, -26, -26, -26, -26, -26, -26, -26, -26, -26, -26,
        // State 2
        -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7,
        // State 3
        0, 0, 0, 23, 24, 0, -25, 0, 0, 0, 0, 0, 0, 0,
        // State 4
        0, -6, 26, -6, -6, 27, -6, 0, 0, 0, 0, 0, 0, 0,
        // State 5
        -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17, -17,
        // State 6
        17, 0, 0, 0, 0, 0, 0, 0, 18, -28, 0, 19, 20, 21,
        // State 7
        0, 0, -22, -22, -22, -22, -22, 30, 0, 0, 31, 0, 0, 0,
        // State 8
        -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29, -29,
        // State 9
        -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
        // State 10
        -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20, -20,
        // State 11
        0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0,
        // State 12
        -32, -32, -32, -32, -32, -32, -32, -32, -32, -32, -32, -32, -32, -32,
        // State 13
        -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
        // State 14
        -21, -21, -21, -21, -21, -21, -21, -21, -21, -21, -21, -21, -21, -21,
        // State 15
        -33, -33, -33, -33, -33, -33, -33, -33, -33, -33, -33, -33, -33, -33,
        // State 16
        17, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 19, 20, 21,
        // State 17
        17, 0, 0, 0, 0, 0, 0, 0, 18, -27, 0, 19, 20, 21,
        // State 18
        -24, -24, -24, -24, -24, -24, -24, -24, -24, -24, -24, -24, -24, -24,
        // State 19
        -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19, -19,
        // State 20
        -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31, -31,
        // State 21
        17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 20, 21,
        // State 22
        -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8, -8,
        // State 23
        -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9,
        // State 24
        17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 20, 21,
        // State 25
        -12, -12, -12, -12, -12, -12, -12, -12, -12, -12, -12, -12, -12, -12,
        // State 26
        -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13, -13,
        // State 27
        -18, -18, -18, -18, -18, -18, -18, -18, -18, -18, -18, -18, -18, -18,
        // State 28
        17, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0, 19, 20, 21,
        // State 29
        -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
        // State 30
        -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2,
        // State 31
        -14, -14, -14, -14, -14, -14, -14, -14, -14, -14, -14, -14, -14, -14,
        // State 32
        0, 39, 0, 23, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 33
        -22, -22, -22, -22, -22, -22, -22, -22, -22, -22, -22, -22, -22, -22,
        // State 34
        0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0,
        // State 35
        0, -23, 26, -23, -23, 27, -23, 0, 0, 0, 0, 0, 0, 0,
        // State 36
        -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,
        // State 37
        0, 0, 0, 23, 24, 0, -1, 0, 0, 0, 0, 0, 0, 0,
        // State 38
        -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30,
        // State 39
        -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,
    ];
    const __EOF_ACTION: &'static [i32] = &[
        -27,
        -26,
        -7,
        0,
        0,
        -17,
        -28,
        0,
        -29,
        -5,
        -20,
        0,
        -32,
        -11,
        -21,
        -33,
        0,
        0,
        -24,
        -19,
        -31,
        0,
        -8,
        -9,
        0,
        -12,
        -13,
        -18,
        0,
        -3,
        -2,
        -14,
        0,
        -22,
        0,
        0,
        -10,
        0,
        -30,
        -4,
    ];
    const __GOTO: &'static [i32] = &[
        // State 0
        2, 0, 3, 4, 0, 5, 0, 6, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0,
        // State 1
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 2
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 3
        0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 4
        0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 5
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 6
        2, 0, 3, 4, 0, 5, 0, 28, 0, 0, 8, 9, 10, 11, 12, 0, 14, 15, 0, 0,
        // State 7
        0, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 8
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 9
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 10
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 11
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 12
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 13
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 14
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 15
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 16
        0, 0, 3, 33, 0, 5, 0, 0, 0, 0, 34, 9, 10, 11, 0, 0, 14, 15, 0, 0,
        // State 17
        2, 0, 3, 4, 0, 5, 0, 6, 0, 7, 8, 9, 10, 11, 12, 35, 14, 15, 0, 0,
        // State 18
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 19
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 20
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 21
        0, 0, 0, 0, 0, 36, 0, 0, 0, 0, 34, 9, 0, 11, 0, 0, 14, 15, 0, 0,
        // State 22
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 23
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 24
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 9, 0, 11, 0, 0, 37, 15, 0, 0,
        // State 25
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 26
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 27
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 28
        0, 0, 3, 38, 0, 5, 0, 0, 0, 0, 34, 9, 10, 11, 0, 0, 14, 15, 0, 0,
        // State 29
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 30
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 31
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 32
        0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 33
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 34
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 35
        0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 36
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 37
        0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 38
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // State 39
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    fn __expected_tokens(__state: usize) -> Vec<::std::string::String> {
        const __TERMINAL: &'static [&'static str] = &[
            r###""(""###,
            r###"")""###,
            r###""*""###,
            r###""+""###,
            r###""-""###,
            r###""/""###,
            r###"";""###,
            r###""=""###,
            r###""{""###,
            r###""}""###,
            r###""~""###,
            r###"r#"[0-9]+"#"###,
            r###"r#"[a-zA-Z][a-zA-Z0-9]*"#"###,
            r###"r#"\\x22[^\\x22]*\\x22"#"###,
        ];
        __ACTION[(__state * 14)..].iter().zip(__TERMINAL).filter_map(|(&state, terminal)| {
            if state == 0 {
                None
            } else {
                Some(terminal.to_string())
            }
        }).collect()
    }
    pub fn parse_Unit<
        'input,
    >(
        input: &'input str,
    ) -> Result<Box<Expr>, __lalrpop_util::ParseError<usize, (usize, &'input str), ()>>
    {
        let mut __tokens = super::__intern_token::__Matcher::new(input);
        let mut __states = vec![0_i32];
        let mut __symbols = vec![];
        let mut __integer;
        let mut __lookahead;
        let mut __last_location = Default::default();
        '__shift: loop {
            __lookahead = match __tokens.next() {
                Some(Ok(v)) => v,
                None => break '__shift,
                Some(Err(e)) => return Err(e),
            };
            __last_location = __lookahead.2.clone();
            __integer = match __lookahead.1 {
                (3, _) if true => 0,
                (4, _) if true => 1,
                (5, _) if true => 2,
                (6, _) if true => 3,
                (7, _) if true => 4,
                (8, _) if true => 5,
                (9, _) if true => 6,
                (10, _) if true => 7,
                (11, _) if true => 8,
                (12, _) if true => 9,
                (13, _) if true => 10,
                (0, _) if true => 11,
                (1, _) if true => 12,
                (2, _) if true => 13,
                _ => {
                    let __state = *__states.last().unwrap() as usize;
                    let __error = __lalrpop_util::ParseError::UnrecognizedToken {
                        token: Some(__lookahead),
                        expected: __expected_tokens(__state),
                    };
                    return Err(__error);
                }
            };
            '__inner: loop {
                let __state = *__states.last().unwrap() as usize;
                let __action = __ACTION[__state * 14 + __integer];
                if __action > 0 {
                    let __symbol = match __integer {
                        0 => match __lookahead.1 {
                            (3, __tok0) => __Symbol::Term_22_28_22((__tok0)),
                            _ => unreachable!(),
                        },
                        1 => match __lookahead.1 {
                            (4, __tok0) => __Symbol::Term_22_29_22((__tok0)),
                            _ => unreachable!(),
                        },
                        2 => match __lookahead.1 {
                            (5, __tok0) => __Symbol::Term_22_2a_22((__tok0)),
                            _ => unreachable!(),
                        },
                        3 => match __lookahead.1 {
                            (6, __tok0) => __Symbol::Term_22_2b_22((__tok0)),
                            _ => unreachable!(),
                        },
                        4 => match __lookahead.1 {
                            (7, __tok0) => __Symbol::Term_22_2d_22((__tok0)),
                            _ => unreachable!(),
                        },
                        5 => match __lookahead.1 {
                            (8, __tok0) => __Symbol::Term_22_2f_22((__tok0)),
                            _ => unreachable!(),
                        },
                        6 => match __lookahead.1 {
                            (9, __tok0) => __Symbol::Term_22_3b_22((__tok0)),
                            _ => unreachable!(),
                        },
                        7 => match __lookahead.1 {
                            (10, __tok0) => __Symbol::Term_22_3d_22((__tok0)),
                            _ => unreachable!(),
                        },
                        8 => match __lookahead.1 {
                            (11, __tok0) => __Symbol::Term_22_7b_22((__tok0)),
                            _ => unreachable!(),
                        },
                        9 => match __lookahead.1 {
                            (12, __tok0) => __Symbol::Term_22_7d_22((__tok0)),
                            _ => unreachable!(),
                        },
                        10 => match __lookahead.1 {
                            (13, __tok0) => __Symbol::Term_22_7e_22((__tok0)),
                            _ => unreachable!(),
                        },
                        11 => match __lookahead.1 {
                            (0, __tok0) => __Symbol::Termr_23_22_5b0_2d9_5d_2b_22_23((__tok0)),
                            _ => unreachable!(),
                        },
                        12 => match __lookahead.1 {
                            (1, __tok0) => __Symbol::Termr_23_22_5ba_2dzA_2dZ_5d_5ba_2dzA_2dZ0_2d9_5d_2a_22_23((__tok0)),
                            _ => unreachable!(),
                        },
                        13 => match __lookahead.1 {
                            (2, __tok0) => __Symbol::Termr_23_22_5c_5cx22_5b_5e_5c_5cx22_5d_2a_5c_5cx22_22_23((__tok0)),
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    };
                    __states.push(__action - 1);
                    __symbols.push((__lookahead.0, __symbol, __lookahead.2));
                    continue '__shift;
                } else if __action < 0 {
                    if let Some(r) = __reduce(input, __action, Some(&__lookahead.0), &mut __states, &mut __symbols, ::std::marker::PhantomData::<()>) {
                        return r;
                    }
                } else {
                    let __state = *__states.last().unwrap() as usize;
                    let __error = __lalrpop_util::ParseError::UnrecognizedToken {
                        token: Some(__lookahead),
                        expected: __expected_tokens(__state),
                    };
                    return Err(__error)
                }
            }
        }
        loop {
            let __state = *__states.last().unwrap() as usize;
            let __action = __EOF_ACTION[__state];
            if __action < 0 {
                if let Some(r) = __reduce(input, __action, None, &mut __states, &mut __symbols, ::std::marker::PhantomData::<()>) {
                    return r;
                }
            } else {
                let __state = *__states.last().unwrap() as usize;
                let __error = __lalrpop_util::ParseError::UnrecognizedToken {
                    token: None,
                    expected: __expected_tokens(__state),
                };
                return Err(__error);
            }
        }
    }
    pub fn __reduce<
        'input,
    >(
        input: &'input str,
        __action: i32,
        __lookahead_start: Option<&usize>,
        __states: &mut ::std::vec::Vec<i32>,
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>,
        _: ::std::marker::PhantomData<()>,
    ) -> Option<Result<Box<Expr>,__lalrpop_util::ParseError<usize, (usize, &'input str), ()>>>
    {
        let __nonterminal = match -__action {
            1 => {
                // Assign = Identifier, AssignOp, Expr => ActionFn(10);
                let __sym2 = __pop_NtExpr(__symbols);
                let __sym1 = __pop_NtAssignOp(__symbols);
                let __sym0 = __pop_NtIdentifier(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym2.2.clone();
                let __nt = super::__action10::<>(input, __sym0, __sym1, __sym2);
                let __states_len = __states.len();
                __states.truncate(__states_len - 3);
                __symbols.push((__start, __Symbol::NtAssign(__nt), __end));
                0
            }
            2 => {
                // AssignOp = "~" => ActionFn(11);
                let __sym0 = __pop_Term_22_7e_22(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action11::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtAssignOp(__nt), __end));
                1
            }
            3 => {
                // AssignOp = "=" => ActionFn(12);
                let __sym0 = __pop_Term_22_3d_22(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action12::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtAssignOp(__nt), __end));
                1
            }
            4 => {
                // Block = "{", Statements, "}" => ActionFn(2);
                let __sym2 = __pop_Term_22_7d_22(__symbols);
                let __sym1 = __pop_NtStatements(__symbols);
                let __sym0 = __pop_Term_22_7b_22(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym2.2.clone();
                let __nt = super::__action2::<>(input, __sym0, __sym1, __sym2);
                let __states_len = __states.len();
                __states.truncate(__states_len - 3);
                __symbols.push((__start, __Symbol::NtBlock(__nt), __end));
                2
            }
            5 => {
                // Expr = MathExpr => ActionFn(7);
                let __sym0 = __pop_NtMathExpr(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action7::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtExpr(__nt), __end));
                3
            }
            6 => {
                // Expr = Factor => ActionFn(8);
                let __sym0 = __pop_NtFactor(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action8::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtExpr(__nt), __end));
                3
            }
            7 => {
                // Expr = Block => ActionFn(9);
                let __sym0 = __pop_NtBlock(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action9::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtExpr(__nt), __end));
                3
            }
            8 => {
                // ExprOp = "+" => ActionFn(14);
                let __sym0 = __pop_Term_22_2b_22(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action14::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtExprOp(__nt), __end));
                4
            }
            9 => {
                // ExprOp = "-" => ActionFn(15);
                let __sym0 = __pop_Term_22_2d_22(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action15::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtExprOp(__nt), __end));
                4
            }
            10 => {
                // Factor = Factor, FactorOp, SubExpr => ActionFn(16);
                let __sym2 = __pop_NtSubExpr(__symbols);
                let __sym1 = __pop_NtFactorOp(__symbols);
                let __sym0 = __pop_NtFactor(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym2.2.clone();
                let __nt = super::__action16::<>(input, __sym0, __sym1, __sym2);
                let __states_len = __states.len();
                __states.truncate(__states_len - 3);
                __symbols.push((__start, __Symbol::NtFactor(__nt), __end));
                5
            }
            11 => {
                // Factor = SubExpr => ActionFn(17);
                let __sym0 = __pop_NtSubExpr(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action17::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtFactor(__nt), __end));
                5
            }
            12 => {
                // FactorOp = "*" => ActionFn(18);
                let __sym0 = __pop_Term_22_2a_22(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action18::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtFactorOp(__nt), __end));
                6
            }
            13 => {
                // FactorOp = "/" => ActionFn(19);
                let __sym0 = __pop_Term_22_2f_22(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action19::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtFactorOp(__nt), __end));
                6
            }
            14 => {
                // FullStatement = Statement, ";" => ActionFn(4);
                let __sym1 = __pop_Term_22_3b_22(__symbols);
                let __sym0 = __pop_NtStatement(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym1.2.clone();
                let __nt = super::__action4::<>(input, __sym0, __sym1);
                let __states_len = __states.len();
                __states.truncate(__states_len - 2);
                __symbols.push((__start, __Symbol::NtFullStatement(__nt), __end));
                7
            }
            15 => {
                // FullStatement* =  => ActionFn(28);
                let __start = __symbols.last().map(|s| s.2.clone()).unwrap_or_default();
                let __end = __lookahead_start.cloned().unwrap_or_else(|| __start.clone());
                let __nt = super::__action28::<>(input, &__start, &__end);
                let __states_len = __states.len();
                __states.truncate(__states_len - 0);
                __symbols.push((__start, __Symbol::NtFullStatement_2a(__nt), __end));
                8
            }
            16 => {
                // FullStatement* = FullStatement+ => ActionFn(29);
                let __sym0 = __pop_NtFullStatement_2b(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action29::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtFullStatement_2a(__nt), __end));
                8
            }
            17 => {
                // FullStatement+ = FullStatement => ActionFn(30);
                let __sym0 = __pop_NtFullStatement(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action30::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtFullStatement_2b(__nt), __end));
                9
            }
            18 => {
                // FullStatement+ = FullStatement+, FullStatement => ActionFn(31);
                let __sym1 = __pop_NtFullStatement(__symbols);
                let __sym0 = __pop_NtFullStatement_2b(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym1.2.clone();
                let __nt = super::__action31::<>(input, __sym0, __sym1);
                let __states_len = __states.len();
                __states.truncate(__states_len - 2);
                __symbols.push((__start, __Symbol::NtFullStatement_2b(__nt), __end));
                9
            }
            19 => {
                // Identifier = r#"[a-zA-Z][a-zA-Z0-9]*"# => ActionFn(27);
                let __sym0 = __pop_Termr_23_22_5ba_2dzA_2dZ_5d_5ba_2dzA_2dZ0_2d9_5d_2a_22_23(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action27::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtIdentifier(__nt), __end));
                10
            }
            20 => {
                // Literal = Num => ActionFn(22);
                let __sym0 = __pop_NtNum(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action22::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtLiteral(__nt), __end));
                11
            }
            21 => {
                // Literal = Text => ActionFn(23);
                let __sym0 = __pop_NtText(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action23::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtLiteral(__nt), __end));
                11
            }
            22 => {
                // Literal = Identifier => ActionFn(24);
                let __sym0 = __pop_NtIdentifier(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action24::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtLiteral(__nt), __end));
                11
            }
            23 => {
                // MathExpr = Expr, ExprOp, Factor => ActionFn(13);
                let __sym2 = __pop_NtFactor(__symbols);
                let __sym1 = __pop_NtExprOp(__symbols);
                let __sym0 = __pop_NtExpr(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym2.2.clone();
                let __nt = super::__action13::<>(input, __sym0, __sym1, __sym2);
                let __states_len = __states.len();
                __states.truncate(__states_len - 3);
                __symbols.push((__start, __Symbol::NtMathExpr(__nt), __end));
                12
            }
            24 => {
                // Num = r#"[0-9]+"# => ActionFn(25);
                let __sym0 = __pop_Termr_23_22_5b0_2d9_5d_2b_22_23(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action25::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtNum(__nt), __end));
                13
            }
            25 => {
                // Statement = Expr => ActionFn(5);
                let __sym0 = __pop_NtExpr(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action5::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtStatement(__nt), __end));
                14
            }
            26 => {
                // Statement = Assign => ActionFn(6);
                let __sym0 = __pop_NtAssign(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action6::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtStatement(__nt), __end));
                14
            }
            27 => {
                // Statements =  => ActionFn(32);
                let __start = __symbols.last().map(|s| s.2.clone()).unwrap_or_default();
                let __end = __lookahead_start.cloned().unwrap_or_else(|| __start.clone());
                let __nt = super::__action32::<>(input, &__start, &__end);
                let __states_len = __states.len();
                __states.truncate(__states_len - 0);
                __symbols.push((__start, __Symbol::NtStatements(__nt), __end));
                15
            }
            28 => {
                // Statements = FullStatement+ => ActionFn(33);
                let __sym0 = __pop_NtFullStatement_2b(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action33::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtStatements(__nt), __end));
                15
            }
            29 => {
                // SubExpr = Literal => ActionFn(20);
                let __sym0 = __pop_NtLiteral(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action20::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtSubExpr(__nt), __end));
                16
            }
            30 => {
                // SubExpr = "(", Expr, ")" => ActionFn(21);
                let __sym2 = __pop_Term_22_29_22(__symbols);
                let __sym1 = __pop_NtExpr(__symbols);
                let __sym0 = __pop_Term_22_28_22(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym2.2.clone();
                let __nt = super::__action21::<>(input, __sym0, __sym1, __sym2);
                let __states_len = __states.len();
                __states.truncate(__states_len - 3);
                __symbols.push((__start, __Symbol::NtSubExpr(__nt), __end));
                16
            }
            31 => {
                // Text = r#"\\x22[^\\x22]*\\x22"# => ActionFn(26);
                let __sym0 = __pop_Termr_23_22_5c_5cx22_5b_5e_5c_5cx22_5d_2a_5c_5cx22_22_23(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action26::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtText(__nt), __end));
                17
            }
            32 => {
                // Unit = Statements => ActionFn(1);
                let __sym0 = __pop_NtStatements(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action1::<>(input, __sym0);
                let __states_len = __states.len();
                __states.truncate(__states_len - 1);
                __symbols.push((__start, __Symbol::NtUnit(__nt), __end));
                18
            }
            33 => {
                // __Unit = Unit => ActionFn(0);
                let __sym0 = __pop_NtUnit(__symbols);
                let __start = __sym0.0.clone();
                let __end = __sym0.2.clone();
                let __nt = super::__action0::<>(input, __sym0);
                return Some(Ok(__nt));
            }
            _ => panic!("invalid action code {}", __action)
        };
        let __state = *__states.last().unwrap() as usize;
        let __next_state = __GOTO[__state * 20 + __nonterminal] - 1;
        __states.push(__next_state);
        None
    }
    fn __pop_Term_22_28_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_28_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_29_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_29_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_2a_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_2a_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_2b_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_2b_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_2d_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_2d_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_2f_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_2f_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_3b_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_3b_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_3d_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_3d_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_7b_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_7b_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_7d_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_7d_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Term_22_7e_22<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Term_22_7e_22(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Termr_23_22_5b0_2d9_5d_2b_22_23<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Termr_23_22_5b0_2d9_5d_2b_22_23(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Termr_23_22_5ba_2dzA_2dZ_5d_5ba_2dzA_2dZ0_2d9_5d_2a_22_23<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Termr_23_22_5ba_2dzA_2dZ_5d_5ba_2dzA_2dZ0_2d9_5d_2a_22_23(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Termr_23_22_5c_5cx22_5b_5e_5c_5cx22_5d_2a_5c_5cx22_22_23<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, &'input str, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Termr_23_22_5c_5cx22_5b_5e_5c_5cx22_5d_2a_5c_5cx22_22_23(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtAssign<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Box<Expr>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtAssign(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtAssignOp<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Opcode, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtAssignOp(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtBlock<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Box<Expr>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtBlock(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtExpr<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Box<Expr>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtExpr(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtExprOp<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Opcode, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtExprOp(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtFactor<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Box<Expr>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtFactor(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtFactorOp<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Opcode, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtFactorOp(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtFullStatement<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Box<Expr>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtFullStatement(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtFullStatement_2a<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, ::std::vec::Vec<Box<Expr>>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtFullStatement_2a(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtFullStatement_2b<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, ::std::vec::Vec<Box<Expr>>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtFullStatement_2b(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtIdentifier<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, String, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtIdentifier(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtLiteral<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Box<Expr>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtLiteral(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtMathExpr<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Box<Expr>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtMathExpr(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtNum<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Box<Expr>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtNum(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtStatement<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Box<Expr>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtStatement(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtStatements<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Box<Expr>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtStatements(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtSubExpr<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Box<Expr>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtSubExpr(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtText<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Box<Expr>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtText(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_NtUnit<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Box<Expr>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::NtUnit(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
    fn __pop_Nt____Unit<
      'input,
    >(
        __symbols: &mut ::std::vec::Vec<(usize,__Symbol<'input>,usize)>
    ) -> (usize, Box<Expr>, usize) {
        match __symbols.pop().unwrap() {
            (__l, __Symbol::Nt____Unit(__v), __r) => (__l, __v, __r),
            _ => panic!("symbol type mismatch")
        }
    }
}
pub use self::__parse__Unit::parse_Unit;
mod __intern_token {
    #![allow(unused_imports)]
    use std::str::FromStr;
    use ast::{Expr, Opcode};
    extern crate lalrpop_util as __lalrpop_util;
    extern crate regex as __regex;
    pub struct __Matcher<'input> {
        text: &'input str,
        consumed: usize,
        regex_set: __regex::RegexSet,
        regex_vec: Vec<__regex::Regex>,
    }

    impl<'input> __Matcher<'input> {
        pub fn new(s: &'input str) -> __Matcher<'input> {
            let __strs: &[&str] = &[
                "^(?u:[0-9])+",
                "^(?u:[A-Za-z])(?u:[0-9A-Za-z])*",
                "^(?u:\")(?u:[\u{0}-!\\#-\u{10ffff}])*(?u:\")",
                "^(?u:\\()",
                "^(?u:\\))",
                "^(?u:\\*)",
                "^(?u:\\+)",
                "^(?u:\\-)",
                "^(?u:/)",
                "^(?u:;)",
                "^(?u:=)",
                "^(?u:\\{)",
                "^(?u:\\})",
                "^(?u:\\~)",
            ];
            let __regex_set = __regex::RegexSet::new(__strs).unwrap();
            let __regex_vec = vec![
                __regex::Regex::new("^(?u:[0-9])+").unwrap(),
                __regex::Regex::new("^(?u:[A-Za-z])(?u:[0-9A-Za-z])*").unwrap(),
                __regex::Regex::new("^(?u:\")(?u:[\u{0}-!\\#-\u{10ffff}])*(?u:\")").unwrap(),
                __regex::Regex::new("^(?u:\\()").unwrap(),
                __regex::Regex::new("^(?u:\\))").unwrap(),
                __regex::Regex::new("^(?u:\\*)").unwrap(),
                __regex::Regex::new("^(?u:\\+)").unwrap(),
                __regex::Regex::new("^(?u:\\-)").unwrap(),
                __regex::Regex::new("^(?u:/)").unwrap(),
                __regex::Regex::new("^(?u:;)").unwrap(),
                __regex::Regex::new("^(?u:=)").unwrap(),
                __regex::Regex::new("^(?u:\\{)").unwrap(),
                __regex::Regex::new("^(?u:\\})").unwrap(),
                __regex::Regex::new("^(?u:\\~)").unwrap(),
            ];
            __Matcher {
                text: s,
                consumed: 0,
                regex_set: __regex_set,
                regex_vec: __regex_vec,
            }
        }
    }

    impl<'input> Iterator for __Matcher<'input> {
        type Item = Result<(usize, (usize, &'input str), usize), __lalrpop_util::ParseError<usize,(usize, &'input str),()>>;

        fn next(&mut self) -> Option<Self::Item> {
            let __text = self.text.trim_left();
            let __whitespace = self.text.len() - __text.len();
            let __start_offset = self.consumed + __whitespace;
            if __text.is_empty() {
                self.text = __text;
                self.consumed = __start_offset;
                None
            } else {
                let __matches = self.regex_set.matches(__text);
                if !__matches.matched_any() {
                    Some(Err(__lalrpop_util::ParseError::InvalidToken {
                        location: __start_offset,
                    }))
                } else {
                    let mut __longest_match = 0;
                    let mut __index = 0;
                    for __i in 0 .. 14 {
                        if __matches.matched(__i) {
                            let __match = self.regex_vec[__i].find(__text).unwrap();
                            let __len = __match.end();
                            if __len >= __longest_match {
                                __longest_match = __len;
                                __index = __i;
                            }
                        }
                    }
                    let __result = &__text[..__longest_match];
                    let __remaining = &__text[__longest_match..];
                    let __end_offset = __start_offset + __longest_match;
                    self.text = __remaining;
                    self.consumed = __end_offset;
                    Some(Ok((__start_offset, (__index, __result), __end_offset)))
                }
            }
        }
    }
}

#[allow(unused_variables)]
fn __action0<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
) -> Box<Expr>
{
    (__0)
}

#[allow(unused_variables)]
fn __action1<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
) -> Box<Expr>
{
    (__0)
}

#[allow(unused_variables)]
fn __action2<
    'input,
>(
    input: &'input str,
    (_, _, _): (usize, &'input str, usize),
    (_, __0, _): (usize, Box<Expr>, usize),
    (_, _, _): (usize, &'input str, usize),
) -> Box<Expr>
{
    (__0)
}

#[allow(unused_variables)]
fn __action3<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, ::std::vec::Vec<Box<Expr>>, usize),
) -> Box<Expr>
{
    Box::new(Expr::List(Vec::from(__0)))
}

#[allow(unused_variables)]
fn __action4<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
    (_, _, _): (usize, &'input str, usize),
) -> Box<Expr>
{
    (__0)
}

#[allow(unused_variables)]
fn __action5<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
) -> Box<Expr>
{
    (__0)
}

#[allow(unused_variables)]
fn __action6<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
) -> Box<Expr>
{
    (__0)
}

#[allow(unused_variables)]
fn __action7<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
) -> Box<Expr>
{
    (__0)
}

#[allow(unused_variables)]
fn __action8<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
) -> Box<Expr>
{
    (__0)
}

#[allow(unused_variables)]
fn __action9<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
) -> Box<Expr>
{
    (__0)
}

#[allow(unused_variables)]
fn __action10<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, String, usize),
    (_, __1, _): (usize, Opcode, usize),
    (_, __2, _): (usize, Box<Expr>, usize),
) -> Box<Expr>
{
    Box::new(Expr::Assign(__0, __1, __2))
}

#[allow(unused_variables)]
fn __action11<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, &'input str, usize),
) -> Opcode
{
    Opcode::Ref
}

#[allow(unused_variables)]
fn __action12<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, &'input str, usize),
) -> Opcode
{
    Opcode::Assign
}

#[allow(unused_variables)]
fn __action13<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
    (_, __1, _): (usize, Opcode, usize),
    (_, __2, _): (usize, Box<Expr>, usize),
) -> Box<Expr>
{
    Box::new(Expr::Op(__0, __1, __2))
}

#[allow(unused_variables)]
fn __action14<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, &'input str, usize),
) -> Opcode
{
    Opcode::Add
}

#[allow(unused_variables)]
fn __action15<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, &'input str, usize),
) -> Opcode
{
    Opcode::Sub
}

#[allow(unused_variables)]
fn __action16<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
    (_, __1, _): (usize, Opcode, usize),
    (_, __2, _): (usize, Box<Expr>, usize),
) -> Box<Expr>
{
    Box::new(Expr::Op(__0, __1, __2))
}

#[allow(unused_variables)]
fn __action17<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
) -> Box<Expr>
{
    (__0)
}

#[allow(unused_variables)]
fn __action18<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, &'input str, usize),
) -> Opcode
{
    Opcode::Mul
}

#[allow(unused_variables)]
fn __action19<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, &'input str, usize),
) -> Opcode
{
    Opcode::Div
}

#[allow(unused_variables)]
fn __action20<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
) -> Box<Expr>
{
    (__0)
}

#[allow(unused_variables)]
fn __action21<
    'input,
>(
    input: &'input str,
    (_, _, _): (usize, &'input str, usize),
    (_, __0, _): (usize, Box<Expr>, usize),
    (_, _, _): (usize, &'input str, usize),
) -> Box<Expr>
{
    (__0)
}

#[allow(unused_variables)]
fn __action22<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
) -> Box<Expr>
{
    (__0)
}

#[allow(unused_variables)]
fn __action23<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
) -> Box<Expr>
{
    (__0)
}

#[allow(unused_variables)]
fn __action24<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, String, usize),
) -> Box<Expr>
{
    Box::new(Expr::Id(__0))
}

#[allow(unused_variables)]
fn __action25<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, &'input str, usize),
) -> Box<Expr>
{
    Box::new(Expr::Number(f64::from_str(__0).unwrap()))
}

#[allow(unused_variables)]
fn __action26<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, &'input str, usize),
) -> Box<Expr>
{
    Box::new(Expr::Text(__0.to_string()))
}

#[allow(unused_variables)]
fn __action27<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, &'input str, usize),
) -> String
{
    __0.to_string()
}

#[allow(unused_variables)]
fn __action28<
    'input,
>(
    input: &'input str,
    __lookbehind: &usize,
    __lookahead: &usize,
) -> ::std::vec::Vec<Box<Expr>>
{
    vec![]
}

#[allow(unused_variables)]
fn __action29<
    'input,
>(
    input: &'input str,
    (_, v, _): (usize, ::std::vec::Vec<Box<Expr>>, usize),
) -> ::std::vec::Vec<Box<Expr>>
{
    v
}

#[allow(unused_variables)]
fn __action30<
    'input,
>(
    input: &'input str,
    (_, __0, _): (usize, Box<Expr>, usize),
) -> ::std::vec::Vec<Box<Expr>>
{
    vec![__0]
}

#[allow(unused_variables)]
fn __action31<
    'input,
>(
    input: &'input str,
    (_, v, _): (usize, ::std::vec::Vec<Box<Expr>>, usize),
    (_, e, _): (usize, Box<Expr>, usize),
) -> ::std::vec::Vec<Box<Expr>>
{
    { let mut v = v; v.push(e); v }
}

#[allow(unused_variables)]
fn __action32<
    'input,
>(
    input: &'input str,
    __lookbehind: &usize,
    __lookahead: &usize,
) -> Box<Expr>
{
    let __start0 = __lookbehind.clone();
    let __end0 = __lookahead.clone();
    let __temp0 = __action28(
        input,
        &__start0,
        &__end0,
    );
    let __temp0 = (__start0, __temp0, __end0);
    __action3(
        input,
        __temp0,
    )
}

#[allow(unused_variables)]
fn __action33<
    'input,
>(
    input: &'input str,
    __0: (usize, ::std::vec::Vec<Box<Expr>>, usize),
) -> Box<Expr>
{
    let __start0 = __0.0.clone();
    let __end0 = __0.2.clone();
    let __temp0 = __action29(
        input,
        __0,
    );
    let __temp0 = (__start0, __temp0, __end0);
    __action3(
        input,
        __temp0,
    )
}

pub trait __ToTriple<'input, > {
    type Error;
    fn to_triple(value: Self) -> Result<(usize,(usize, &'input str),usize),Self::Error>;
}

impl<'input, > __ToTriple<'input, > for (usize, (usize, &'input str), usize) {
    type Error = ();
    fn to_triple(value: Self) -> Result<(usize,(usize, &'input str),usize),()> {
        Ok(value)
    }
}
impl<'input, > __ToTriple<'input, > for Result<(usize, (usize, &'input str), usize),()> {
    type Error = ();
    fn to_triple(value: Self) -> Result<(usize,(usize, &'input str),usize),()> {
        value
    }
}
