# The Noise Programming Language

Noiselang is a expression based, probabilitic language where variables does not take exact values
but a probability distribution.

Noise extracts away the details of the simulation and allows mathematicians to write idiomatic code to deal with random variables.

Monte carlo simulations, queue theory simulations are trivial with Noise.


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
Monte carlo simulation
```
X ~ unif(-1, 1) ** 2
Y ~ unif(-1, 1) ** 2

C === "fall inside circle"
C = X + Y < 1

P(C) // prints 3.14
```

### Dice

```
Dice ~ unif(1, 6)

X === "gettting 4 with a dice"
X = Dice == 4
explain(P(X)) // "probability of gettting 4 with a dice is 1/6"
p

Y === "probability of getting 4, 10 times in a row"
Y = P(X)**10
```