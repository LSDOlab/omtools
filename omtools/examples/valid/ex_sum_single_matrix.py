from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleSingleMatrix(Group):
    def setup(self):
        n = 3
        m = 6

        # Declare a matrix of shape 3x6 as input
        M1 = self.declare_input('M1', val=np.arange(n * m).reshape((n, m)))

        # Output the sum of all the elements of the tensor T1
        self.register_output('single_matrix_sum', ot.sum(M1))


prob = Problem()
prob.model = ExampleSingleMatrix()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('M1', prob['M1'].shape)
print(prob['M1'])
print('single_matrix_sum', prob['single_matrix_sum'].shape)
print(prob['single_matrix_sum'])
