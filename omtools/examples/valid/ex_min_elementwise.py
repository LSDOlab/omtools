from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleElementwise(Group):
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
        self.register_output('ElementwiseMin', ot.min(tensor1, tensor2))


prob = Problem()
prob.model = ExampleElementwise()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('tensor1', prob['tensor1'].shape)
print(prob['tensor1'])
print('tensor2', prob['tensor2'].shape)
print(prob['tensor2'])
print('ElementwiseMin', prob['ElementwiseMin'].shape)
print(prob['ElementwiseMin'])
