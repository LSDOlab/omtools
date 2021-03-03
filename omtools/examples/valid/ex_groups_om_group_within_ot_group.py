from openmdao.api import Problem
import openmdao.api as om
import omtools.api as ot
import numpy as np


class ExampleOMGroupWithinOTGroup(ot.Group):
    def setup(self):
        # Create independent variable using Omtools
        x1 = self.create_indep_var('x1', val=40)

        # Create subsystem that depends on previously created
        # independent variable
        openmdao_subgroup = om.Group()

        # Declaring and creating variables within the omtools subgroup
        openmdao_subgroup.add_subsystem('ivc',
                                        om.IndepVarComp('x2', val=12),
                                        promotes=['*'])
        openmdao_subgroup.add_subsystem('simple_prod',
                                        om.ExecComp('prod_x1x2 = x1 * x2'),
                                        promotes=['*'])

        self.add_subsystem('openmdao_subgroup',
                           openmdao_subgroup,
                           promotes=['*'])

        # Receiving the value of x2 from the openmdao group
        x2 = self.declare_input('x2')
        # Simple addition in the Omtools group
        y1 = x2 + x1
        self.register_output('y1', y1)


prob = Problem()
prob.model = ExampleOMGroupWithinOTGroup()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('x1', prob['x1'].shape)
print(prob['x1'])
print('x2', prob['x2'].shape)
print(prob['x2'])
print('y1', prob['y1'].shape)
print(prob['y1'])
