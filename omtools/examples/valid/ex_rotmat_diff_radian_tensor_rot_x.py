from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleDiffRadianTensorRotX(Group):
    def setup(self):

        # Shape of a random tensor rotation matrix
        shape = (2, 3, 4)

        num_elements = np.prod(shape)

        # Vector of angles in radians
        angle_val2 = np.repeat(
            np.pi / 3, num_elements) + 2 * np.pi * np.arange(num_elements)

        angle_val2 = angle_val2.reshape(shape)

        # Adding the vector as an input
        angle_tensor = self.declare_input('tensor', val=angle_val2)

        # Rotation in the x-axis for tensor2
        self.register_output('tensor_Rot_x', ot.rotmat(angle_tensor, axis='x'))


prob = Problem()
prob.model = ExampleDiffRadianTensorRotX()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('tensor', prob['tensor'].shape)
print(prob['tensor'])
print('tensor_Rot_x', prob['tensor_Rot_x'].shape)
print(prob['tensor_Rot_x'])
