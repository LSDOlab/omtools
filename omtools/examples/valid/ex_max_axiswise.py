from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleAxiswise(Group):
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


prob = Problem()
prob.model = ExampleAxiswise()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('tensor', prob['tensor'].shape)
print(prob['tensor'])
print('AxiswiseMin', prob['AxiswiseMin'].shape)
print(prob['AxiswiseMin'])
