Expressing Cyclic Relationships with Explicit Outputs
=====================================================

We can also define a value in terms of itself.
In the example below, consider three fixed point iterations.
Each of these fixed point iterations is declared within its own
subsystem so that a solver may be assigned to it.
Note that variables are not rpomoted so that we can use the same name to
refer to different variables, depending on which subsystem they belong
to.

.. code-block:: python

  from openmdao.api import NonlinearBlockGS, ScipyKrylov
  
  import omtools.api as ot
  from omtools.api import Group
  from omtools.core.expression import Expression
  from openmdao.api import Problem
  import numpy as np
  
  
  class Example(Group):
      def setup(self):
          # x == (3 + x - 2 * x**2)**(1 / 4)
          group = Group()
          x = group.create_output('x')
          x.define((3 + x - 2 * x**2)**(1 / 4))
          group.nonlinear_solver = NonlinearBlockGS(maxiter=100)
          self.add_subsystem('cycle_1', group)
  
          # x == ((x + 3 - x**4) / 2)**(1 / 4)
          group = Group()
          x = group.create_output('x')
          x.define(((x + 3 - x**4) / 2)**(1 / 4))
          group.nonlinear_solver = NonlinearBlockGS(maxiter=100)
          self.add_subsystem('cycle_2', group)
  
          # x == 0.5 * x
          group = Group()
          x = group.create_output('x')
          x.define(0.5 * x)
          group.nonlinear_solver = NonlinearBlockGS(maxiter=100)
          self.add_subsystem('cycle_3', group)
  
  
  prob = Problem()
  prob.model = Example()
  prob.setup(force_alloc_complex=True)
  # Warm start
  prob.set_val('cycle_3.x', -10)
  prob.run_model()
  
  # embed-module-print not working for displaying values in docs
  print('cycle_1.x', prob['cycle_1.x'])
  print('cycle_2.x', prob['cycle_2.x'])
  print('cycle_3.x', prob['cycle_3.x'])
  

.. embed-n2 ::
  ../examples/ex_explicit_cycles.py
