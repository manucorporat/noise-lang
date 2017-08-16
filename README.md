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
```
X ~ unif(-1, 1) ** 2
Y ~ unif(-1, 1) ** 2
P(X + Y < 1)
```



