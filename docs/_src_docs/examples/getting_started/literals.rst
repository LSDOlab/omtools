Working with Literal Values
===========================

This example shows how to create a very simple expression.
First, the user declares an input with the ``Group.declare_input``
method.
Then, the user forms an expression from the inputs and registers the
output with ``Group.register_output``.

.. jupyter-execute::
  ../../../../omtools/examples/valid/ex_explicit_literals.py

Below, we see how ``omtools`` directs ``OpenMDAO`` to construct a
``Component`` object for each operation.

.. embed-n2 ::
  ../omtools/examples/valid/ex_explicit_literals.py
