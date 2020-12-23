Working with Literal Values
===========================

This example shows how to create a very simple expression.
First, the user declares an input with the ``Group.declare_input``
method.
Then, the user forms an expression from the inputs and registers the
output with ``Group.register_output``.

.. code-block:: python

  from openmdao.api import NonlinearBlockGS, ScipyKrylov
  from openmdao.api import Problem
  
  import omtools.api as ot
  from omtools.api import Group
  from omtools.core.expression import Expression
  
  
  class Example(Group):
      def setup(self):
          x = self.declare_input('x', val=3)
          y = -2 * x**2 + 4 * x + 3
          self.register_output('y', y)
  
  
  prob = Problem()
  prob.model = Example()
  prob.setup(force_alloc_complex=True)
  prob.run_model()
  

Below, we see how ``omtools`` directs ``OpenMDAO`` to construct a
``Component`` object for each operation.

.. embed-n2 ::
  ../omtools/examples/ex_working_with_literal_values.py
