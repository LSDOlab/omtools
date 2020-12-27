Simple Explicit Expressions
---------------------------

In this example, we define multiple expressions and register their
outputs.
Note that the outputs need not be registered in the order in which they
are defined, nor do they necessarily appear in the n2 diagram in the
order in which they are registered.
This is due to the fact taht ``omtools`` automatically rearranges
expressions so that the ``Component`` objects do not have any
unnecessary feedbacks.

.. code-block:: python

  from openmdao.api import NonlinearBlockGS, ScipyKrylov
  from openmdao.api import Problem
  
  import omtools.api as ot
  from omtools.api import Group
  from omtools.core.expression import Expression
  import numpy as np
  
  
  class Example(Group):
      def setup(self):
          # declare inputs with default values
          x1 = self.declare_input('x1', val=2)
          x2 = self.declare_input('x2', val=3)
          x3 = self.declare_input('x3', val=np.arange(7))
  
          # Elementwise addition
          y1 = x2 + x1
          self.register_output('y1', y1)
  
          # Elementwise subtraction
          self.register_output('y2', x2 - x1)
  
          # Elementwise multitplication
          self.register_output('y3', x1 * x2)
  
          # Elementwise division
          self.register_output('y4', x1 / x2)
          self.register_output('y5', x1 / 3)
          self.register_output('y6', 2 / x2)
  
          # Elementwise Power
          y8 = x2**2
          self.register_output('y8', y8)
          self.register_output('y7', x1**2)
  
          # Adding other expressions
          self.register_output('y9', y1 + y8)
  
          # Array of powers
          y10 = x3**(2 * np.ones(7))
          self.register_output('y10', y10)
  
  
  prob = Problem()
  prob.model = Example()
  prob.setup(force_alloc_complex=True)
  prob.run_model()
  print('y10', prob['y10'])
  

The ``Component`` objects added to the model are guaranteed to be
connected such that there are no unnecessary feedbacks, regardless of
the order in which each output is defined or registered.
In this case, since there are no cycles, the n2 diagram is upper
triangular.

.. embed-n2::
  ../omtools/examples/ex_simple_binary_operations.py
