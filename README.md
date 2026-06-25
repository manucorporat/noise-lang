# The Noise Programming Language

Noise is an expression-based, probabilistic language: variables don't hold exact values, they
hold *random variables* (probability distributions). Operators lift over random variables, and
`P(condition)` estimates a probability by simulation — so propagating uncertainty and running
Monte Carlo experiments reads like ordinary math.

**Scope (be honest about it):** Noise today is a **static random-variable algebra + forward
Monte Carlo** tool — excellent for things like estimating π, summing risks, or propagating
uncertainty through a formula. Modeling *dynamic* stochastic systems (queues, Markov chains,
random walks) needs sequential/stateful sampling the language does not have yet; that's a
deliberate future track, not a current capability. See `GOAL.md`, `PLAN.md`, and `AGENT.md`
for the precise state and roadmap.

> **The one rule that surprises everyone:** a name bound with `~` is *one fixed draw* that every
> mention reuses. So `X - X` is exactly `0`, and `Dice + Dice` is `2·Dice` — **not** two dice.
> Independent draws come only from separate `~` bindings (or, later, function calls). See
> "Random variables and sharing" in `LANG.md`.


## Examples
### Assignments
```
X ~ expr(a);
Y = Y+3*(2+3);
U = plot(X)
```

### Everything is a expression
```
X + Y

d = {a=2 b=2 c=a+b} * 10

e = if d > a {
  d
}else{
  a
}
```

### Operators
```
X + Y
X ** Y
X * Y
X / Y

X > 0
x < 0
X == 0
Y != 0
```


### Functions

```
X(y) ~ {
  x = !y;
};

max(x, y) ~ if x > y { x } else { y }
```


### Calculate PI
Monte Carlo simulation. Points fall uniformly in the 2×2 square; the fraction inside the unit
circle is `π/4`, so π is `4 · P(C)`.
```
X ~ unif(-1, 1)
Y ~ unif(-1, 1)

C = X**2 + Y**2 < 1     # "fell inside the circle"

4 * P(C)                // ≈ 3.14   (P(C) alone ≈ 0.785 = π/4)
```

### Dice
A die is **discrete**, so it needs the discrete uniform `unif_int` — with *continuous* `unif(1,6)`,
`P(Dice == 4)` is `0` (a continuous draw never lands exactly on 4).
```
Dice ~ unif_int(1, 6)   # integers 1..=6  (discrete; unif_int is planned, see PLAN.md Phase 3)

P(Dice == 4)            // ≈ 1/6

# "Getting 4 on two dice" needs TWO independent draws — two bindings, not `Dice + Dice`:
A ~ unif_int(1, 6)
B ~ unif_int(1, 6)
P(A == 4 && B == 4)     // ≈ 1/36   (&& is planned, see PLAN.md Phase 3)
```
Note: `P(X)**10` would compute `(1/6)^10` by hand — that's *you* doing the probability, not the
language modeling 10 trials. The modeling form is to sample 10 independent rolls and ask
`P(all ten == 4)` once those features exist.