Implicit Relationships
======================

It is possible to compute outputs implicitly by defining a residual
expression in terms of the output and inputs.

In the first example, we solve a quadratic equation.
This quadratic has two solutions; 1 and 3.
Depending on the starting value of the output variable, OpenMDAO will
find one root or the other.

.. jupyter-execute::
  ../../../omtools/examples/ex_implicit_nonlinear.py

The expressions for the residuals will tell OpenMDAO to construct the
relevant `Component` objects, but they will be part of a `Problem`
within the generated `ImplicitComponent` object.
As a result, the expressions defined above do not translate to
`Component` objects in the outer `Problem` whose model is displayed in
the n2 diagram below.

.. embed-n2::
  ../omtools/examples/ex_implicit_nonlinear.py

For especially complicated problems, where the residual may converge for
multiple solutions, or where the residual is difficult to converge over
some interval, `omtools` provides an API for bracketing solutions.

.. jupyter-execute::
  ../../../omtools/examples/ex_implicit_bracketed_scalar.py

Brackets may also be specified for multidimensional array values

.. jupyter-execute::
  ../../../omtools/examples/ex_implicit_bracketed_array.py
