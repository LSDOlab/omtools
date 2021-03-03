from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleSingleMatrixAlong1(Group):
    def setup(self):
        n = 3
        m = 6

        # Declare a matrix of shape 3x6 as input
        M1 = self.declare_input('M1', val=np.arange(n * m).reshape((n, m)))

        # Output the axiswise sum of matrix M1 along the columns
        self.register_output('single_matrix_sum_along_1', ot.sum(M1,
                                                                 axes=(1, )))


prob = Problem()
prob.model = ExampleSingleMatrixAlong1()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('M1', prob['M1'].shape)
print(prob['M1'])
print('single_matrix_sum_along_1', prob['single_matrix_sum_along_1'].shape)
print(prob['single_matrix_sum_along_1'])
