from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleScalarRotX(Group):
    """
    :param var: scalar
    :param var: scalar_Rot_x
    """
    def setup(self):
        angle_val3 = np.pi / 3

        angle_scalar = self.declare_input('scalar', val=angle_val3)

        # Rotation in the x-axis for scalar
        self.register_output('scalar_Rot_x', ot.rotmat(angle_scalar, axis='x'))


class ExampleScalarRotY(Group):
    """
    :param var: scalar
    :param var: scalar_Rot_y
    """
    def setup(self):
        angle_val3 = np.pi / 3

        angle_scalar = self.declare_input('scalar', val=angle_val3)

        # Rotation in the y-axis for scalar
        self.register_output('scalar_Rot_y', ot.rotmat(angle_scalar, axis='y'))


class ExampleSameRadianTensorRotX(Group):
    """
    :param var: tensor
    :param var: tensor_Rot_x
    """
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


class ExampleDiffRadianTensorRotX(Group):
    """
    :param var: tensor
    :param var: tensor_Rot_x
    """
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
