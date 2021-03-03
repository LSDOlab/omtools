from openmdao.api import Problem
import openmdao.api as om
import omtools.api as ot
import numpy as np


class ExampleOTGroupWithinOMGroup(om.Group):
    def setup(self):
        # Create independent variable using OpenMDAO
        comp = om.IndepVarComp('x1', val=40)
        self.add_subsystem('ivc', comp, promotes=['*'])

        # Create subsystem that depends on previously created
        # independent variable
        omtools_subgroup = ot.Group()

        # Declaring and creating variables within the omtools subgroup
        a = omtools_subgroup.declare_input('x1')
        b = omtools_subgroup.create_indep_var('x2', val=12)
        omtools_subgroup.register_output('prod', a * b)
        self.add_subsystem('omtools_subgroup',
                           omtools_subgroup,
                           promotes=['*'])

        # Simple addition
        self.add_subsystem('simple_addition',
                           om.ExecComp('y1 = x2 + x1'),
                           promotes=['*'])


prob = Problem()
prob.model = ExampleOTGroupWithinOMGroup()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('x1', prob['x1'].shape)
print(prob['x1'])
print('x2', prob['x2'].shape)
print(prob['x2'])
print('y1', prob['y1'].shape)
print(prob['y1'])
