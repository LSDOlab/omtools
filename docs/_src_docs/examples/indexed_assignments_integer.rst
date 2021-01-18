Indexed Assignments (Integer Indices)
=====================================

``omtools`` supports indexed assignments for explicit outputs.
In this example, integer indices are used to concatenate values from
multiple expressions/variables into one variable.

.. jupyter-execute::
  ../../../omtools/examples/valid/ex_indices_integer.py

The result is a component that concatenates values from the inputs
defined by the indices.
Other ``Component`` objects that compute the values that are
concatenated are also constructed.

.. embed-n2 ::
  ../omtools/examples/valid/ex_indices_integer.py
