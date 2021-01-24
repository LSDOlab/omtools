from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleScalarRotY(Group):
    def setup(self):
        angle_val3 = np.pi / 3

        angle_scalar = self.declare_input('scalar', val=angle_val3)

        # Rotation in the y-axis for scalar
        self.register_output('scalar_Rot_y', ot.rotmat(angle_scalar, axis='y'))


prob = Problem()
prob.model = ExampleScalarRotY()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('scalar', prob['scalar'].shape)
print(prob['scalar'])
print('scalar_Rot_y', prob['scalar_Rot_y'].shape)
print(prob['scalar_Rot_y'])
