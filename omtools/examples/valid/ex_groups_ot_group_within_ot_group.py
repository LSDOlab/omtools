from openmdao.api import Problem
import openmdao.api as om
import omtools.api as ot
import numpy as np


class ExampleOTGroupWithinOTGroup(ot.Group):
    """

    """
    def setup(self):
        # Create independent variable
        x1 = self.create_indep_var('x1', val=40)

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

        # Declaring input that will receive its value from the omtools subgroup
        x2 = self.declare_input('x2')

        # Simple addition
        y1 = x2 + x1
        self.register_output('y1', y1)


prob = Problem()
prob.model = ExampleOTGroupWithinOTGroup()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('x1', prob['x1'].shape)
print(prob['x1'])
print('x2', prob['x2'].shape)
print(prob['x2'])
print('y1', prob['y1'].shape)
print(prob['y1'])
