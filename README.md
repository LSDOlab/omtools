# omtools

Tools for building models from expressions in OpenMDAO

## Grammar

Expressions are composed from other expressions using the order of
operations from Python.
Variables are considered nullary expressions.

```haskell
Expr = Expr
     | Unary(Expr) -- ??
     | Expr + Expr | Expr - Expr -- Aunt Sally
     | Expr * Expr | Expr / Expr -- my dear
     | Expr ** Num -- excuse
     | ( Expr ) -- please
     | Var
```
