from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleRotationMatrix(Group):
    def setup(self):

        # Shape of a random tensor rotation matrix
        shape = (2, 3, 4)

        num_elements = np.prod(shape)

        # Tensor of angles in radians
        angle_val1 = np.repeat(np.pi / 3, num_elements).reshape(shape)

        angle_val2 = np.repeat(
            np.pi / 3, num_elements) + 2 * np.pi * np.arange(num_elements)

        angle_val2 = angle_val2.reshape(shape)

        angle_val3 = np.pi / 3

        # Adding the tensor as an input
        angle_tensor1 = self.declare_input('tensor1', val=angle_val1)

        angle_tensor2 = self.declare_input('tensor2', val=angle_val2)

        angle_scalar = self.declare_input('scalar', val=angle_val3)

        # Rotation in the x-axis for scalar
        self.register_output('scalar_Rot_x', ot.rotmat(angle_scalar, axis='x'))

        # Rotation in the y-axis for scalar
        self.register_output('scalar_Rot_y', ot.rotmat(angle_scalar, axis='y'))

        # Rotation in the x-axis for tensor1
        self.register_output('tensor1_Rot_x', ot.rotmat(angle_tensor1,
                                                        axis='x'))

        # Rotation in the x-axis for tensor2
        self.register_output('tensor2_Rot_x', ot.rotmat(angle_tensor2,
                                                        axis='x'))


prob = Problem()
prob.model = ExampleRotationMatrix()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('scalar_Rot_x', prob['scalar_Rot_x'].shape)
print(prob['scalar_Rot_x'])
print('scalar_Rot_y', prob['scalar_Rot_y'].shape)
print(prob['scalar_Rot_y'])
print('tensor1_Rot_x', prob['tensor1_Rot_x'].shape)
print(prob['tensor1_Rot_x'])
print('tensor2_Rot_x', prob['tensor2_Rot_x'].shape)
print(prob['tensor2_Rot_x'])
