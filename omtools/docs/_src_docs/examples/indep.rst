Independent Variables
=====================

Creating an independent variable always registers that variable as an
output, regardless of whether it is used in an expression, or whether
any expression that uses the independent variable is registered as an
output.

This means that all independent variables are available to parent
``System`` objects.

In this example, a single independent variable is created within the
model even though there are no dependencies on the independent variable
within the model.

.. code-block:: python

  from omtools.api import Group
  from openmdao.api import Problem
  
  
  class Example(Group):
      def setup(self):
          z = self.create_indep_var('z', val=10)
  
  
  prob = Problem()
  prob.model = Example()
  prob.setup()
  prob.run_model()
  

.. embed-n2 ::
  ../examples/ex_create_indep_var.py
