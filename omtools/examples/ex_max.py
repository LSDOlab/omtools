from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleScalar(Group):
    """
    :param var: tensor
    :param var: ScalarMin
    """
    def setup(self):
        m = 2
        n = 3
        o = 4
        p = 5
        q = 6

        # Shape of a tensor
        tensor_shape = (m, n, o, p, q)

        num_of_elements = np.prod(tensor_shape)
        # Creating the values of the tensor
        val = np.arange(num_of_elements).reshape(tensor_shape)

        # Declaring the tensor as an input
        ten = self.declare_input('tensor', val=val)

        # Computing the minimum across the entire tensor, returns single value
        self.register_output('ScalarMin', ot.max(ten))


class ExampleAxiswise(Group):
    """
    :param var: tensor
    :param var: AxiswiseMin
    """
    def setup(self):
        m = 2
        n = 3
        o = 4
        p = 5
        q = 6

        # Shape of a tensor
        tensor_shape = (m, n, o, p, q)

        num_of_elements = np.prod(tensor_shape)
        # Creating the values of the tensor
        val = np.arange(num_of_elements).reshape(tensor_shape)

        # Declaring the tensor as an input
        ten = self.declare_input('tensor', val=val)

        # Computing the axiswise minimum on the tensor
        axis = 1
        self.register_output('AxiswiseMin', ot.max(ten, axis=axis))


class ExampleElementwise(Group):
    """
    :param var: tensor1
    :param var: tensor2
    :param var: ElementwiseMin
    """
    def setup(self):

        m = 2
        n = 3
        # Shape of the three tensors is (2,3)
        shape = (m, n)

        # Creating the values for two tensors
        val1 = np.array([[1, 5, -8], [10, -3, -5]])
        val2 = np.array([[2, 6, 9], [-1, 2, 4]])

        # Declaring the two input tensors
        tensor1 = self.declare_input('tensor1', val=val1)
        tensor2 = self.declare_input('tensor2', val=val2)

        # Creating the output for matrix multiplication
        self.register_output('ElementwiseMin', ot.max(tensor1, tensor2))


class ErrorMultiInputsAndAxis(Group):
    def setup(self):
        # Creating the values for two tensors
        val1 = np.array([[1, 5, -8], [10, -3, -5]])
        val2 = np.array([[2, 6, 9], [-1, 2, 4]])

        # Declaring the two input tensors
        tensor1 = self.declare_input('tensor1', val=val1)
        tensor2 = self.declare_input('tensor2', val=val2)

        # Creating the output for matrix multiplication
        self.register_output('ElementwiseMinWithAxis',
                             ot.max(tensor1, tensor2, axis=0))


class ErrorInputsNotSameSize(Group):
    def setup(self):
        # Creating the values for two tensors
        val1 = np.array([[1, 5], [10, -3]])
        val2 = np.array([[2, 6, 9], [-1, 2, 4]])

        # Declaring the two input tensors
        tensor1 = self.declare_input('tensor1', val=val1)
        tensor2 = self.declare_input('tensor2', val=val2)

        # Creating the output for matrix multiplication
        self.register_output('ElementwiseMinWrongSize',
                             ot.max(tensor1, tensor2))
