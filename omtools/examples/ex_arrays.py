from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleMin(Group):
    """
    :param var: ScalarMin
    :param var: ElementwiseMin
    :param var: AxiswiseMin
    """
    def setup(self):
        m = 2
        n = 3
        o = 4
        p = 5
        q = 6
        """
        Scalar and Axis-wise minimum of a tensor across the (1,3) axis
        """

        # Shape of a tensor
        tensor_shape = (m, n, o, p, q)

        num_of_elements = np.prod(tensor_shape)
        # Creating the values of the tensor
        val = np.arange(num_of_elements).reshape(tensor_shape)

        # Declaring the tensor as an input
        ten1 = self.declare_input('ten1', val=val)

        # Computing the axiswise minimum on the tensor
        axis = 1
        self.register_output('AxiswiseMin', ot.min(ten1, axis=axis))

        # Computing the minimum across the entire tensor, returns single value
        self.register_output('ScalarMin', ot.min(ten1))
        """
        Element-wise minimum between three tensors
        """
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
        self.register_output('ElementwiseMin', ot.min(tensor1, tensor2))


class ExampleMax(Group):
    """
    :param var: ScalarMax
    :param var: ElementwiseMax
    :param var: AxiswiseMax
    """
    def setup(self):
        m = 2
        n = 3
        o = 4
        p = 5
        q = 6
        """
        Scalar and Axis-wise maximum of a tensor across the (1,3) axis
        """

        # Shape of a tensor
        tensor_shape = (m, n, o, p, q)

        num_of_elements = np.prod(tensor_shape)
        # Creating the values of the tensor
        val = np.arange(num_of_elements).reshape(tensor_shape)

        # Declaring the tensor as an input
        ten1 = self.declare_input('ten1', val=val)

        # Computing the axiswise minimum on the tensor
        axis = 1
        self.register_output('AxiswiseMax', ot.max(ten1, axis=axis))

        # Computing the minimum across the entire tensor, returns single value
        self.register_output('ScalarMax', ot.max(ten1))
        """
        Element-wise minimum between three tensors
        """
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

        # Creating the output for tensorrix multiplication
        self.register_output('ElementwiseMax', ot.max(tensor1, tensor2))


class ExampleReshape(Group):
    """
    :param var: reshape_tensor2vector
    :param var: reshape_vector2tensor
    """
    def setup(self):
        shape = (2, 3, 4, 5)
        size = 2 * 3 * 4 * 5

        # Both vector or tensors need to be numpy arrays
        tensor = np.arange(size).reshape(shape)
        vector = np.arange(size)

        # in1 is a tensor having shape = (2, 3, 4, 5)
        in1 = self.declare_input('in1', val=tensor)

        # in2 is a vector having a size of 2 * 3 * 4 * 5
        in2 = self.declare_input('in2', val=vector)

        # in1 is being reshaped from shape = (2, 3, 4, 5) to a vector
        # having size = 2 * 3 * 4 * 5
        self.register_output('reshape_tensor2vector',
                             ot.reshape(in1, new_shape=(size, )))

        # in2 is being reshaped from size =  2 * 3 * 4 * 5 to a tenßsor
        # having shape = (2, 3, 4, 5)
        self.register_output('reshape_vector2tensor',
                             ot.reshape(in2, new_shape=shape))


class ExampleReorderAxes(Group):
    """
    :param var: axes_reordered_matrix
    :param var: axes_reordered_tensor
    """
    def setup(self):

        # Declare mat as an input matrix with shape = (4, 2)
        mat = self.declare_input(
            'M1',
            val=np.arange(4 * 2).reshape((4, 2)),
        )

        # Declare tens as an input tensor with shape = (4, 3, 2, 5)
        tens = self.declare_input(
            'T1',
            val=np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2)),
        )

        # Compute the transpose of mat
        self.register_output('axes_reordered_matrix',
                             ot.reorder_axes(mat, 'ij->ji'))

        # Compute a new tensor by reordering axes of tens; reordering is
        # given by 'ijkl->ljki'
        self.register_output('axes_reordered_tensor',
                             ot.reorder_axes(tens, 'ijkl->ljki'))
