from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleSingleTensor(Group):
    def setup(self):
        n = 3
        m = 4
        p = 5
        q = 6

        # Declare a tensor of shape 3x6x7x10 as input
        T1 = self.declare_input('T1',
                                val=np.arange(n * m * p * q).reshape(
                                    (n, m, p, q)))
        # Output the sum of all the elements of the matrix M1
        self.register_output('single_tensor_sum', ot.sum(T1))


prob = Problem()
prob.model = ExampleSingleTensor()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('T1', prob['T1'].shape)
print(prob['T1'])
print('single_tensor_sum', prob['single_tensor_sum'].shape)
print(prob['single_tensor_sum'])
