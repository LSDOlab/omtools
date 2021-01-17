from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleMax(Group):
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


prob = Problem()
prob.model = ExampleMax()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('ScalarMax', prob['ScalarMax'].shape)
print(prob['ScalarMax'])
print('ElementwiseMax', prob['ElementwiseMax'].shape)
print(prob['ElementwiseMax'])
print('AxiswiseMax', prob['AxiswiseMax'].shape)
print(prob['AxiswiseMax'])
