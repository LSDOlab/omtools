from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleSameRadianTensorRotX(Group):
    def setup(self):

        # Shape of a random tensor rotation matrix
        shape = (2, 3, 4)

        num_elements = np.prod(shape)

        # Tensor of angles in radians
        angle_val1 = np.repeat(np.pi / 3, num_elements).reshape(shape)

        # Adding the tensor as an input
        angle_tensor1 = self.declare_input('tensor', val=angle_val1)

        # Rotation in the x-axis for tensor1
        self.register_output('tensor_Rot_x', ot.rotmat(angle_tensor1,
                                                       axis='x'))


prob = Problem()
prob.model = ExampleSameRadianTensorRotX()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('tensor', prob['tensor'].shape)
print(prob['tensor'])
print('tensor_Rot_x', prob['tensor_Rot_x'].shape)
print(prob['tensor_Rot_x'])
