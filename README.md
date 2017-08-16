# The Noice Programming Language

Noicelang is a expression based, probabilitic language where variables does not take exact values
but a probability distribution.

Noice extracts away the details of the simulation and allows mathematicians to write idiomatic code to deal with random variables.

Monte carlo simulations, queue theory simulations are trivial with Noice.


```
X ~ expr(a);
Y = Y+3*(2+3);

U = plot(X)

X + Y
X ** Y

X > 0
x < 0
X == 0
Y != 0


X(y) ~ {
  x = !y;
};


