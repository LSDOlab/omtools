Failing to Register Outputs Results in No Components
----------------------------------------------------

Here a subsystem is added as in the previous example, but no outputs are
registered in the parent ``Group``.

This "dead code" does not lead to OpenMDAO constructing any
``Component`` objects, and no ``Component`` objects appear in the n2
diagram for this model, other than the ``Component`` objects that
correspond to the outputs registered in ``'sys'``.

.. code-block:: python

  from openmdao.api import NonlinearBlockGS, ScipyKrylov
  from openmdao.api import Problem

  import omtools.api as ot
  from omtools.api import Group
  from omtools.core.expression import Expression


  class Example(Group):
      def setup(self):
          group = Group()
          a = group.declare_input('a', val=2)
          b = group.create_indep_var('b', val=12)
          group.register_output('prod', a * b)
          self.add_subsystem('sys', group, promotes=['*'])

          # These expressions do not lead to constructing any Component
          # objects
          x1 = self.declare_input('x1')
          x2 = self.declare_input('x2')
          y1 = x2 + x1
          y2 = x2 - x1
          y3 = x1 * x2
          y4 = x2**2


  prob = Problem()
  prob.model = Example()
  prob.setup(force_alloc_complex=True)
  prob.run_model()


.. embed-n2 ::
  ../omtools/examples/ex_no_registered_outputs.py
