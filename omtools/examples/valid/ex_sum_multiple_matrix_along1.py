from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleMultipleMatrixAlong1(Group):
    def setup(self):
        n = 3
        m = 6

        # Declare a matrix of shape 3x6 as input
        M1 = self.declare_input('M1', val=np.arange(n * m).reshape((n, m)))

        # Declare another matrix of shape 3x6 as input
        M2 = self.declare_input('M2',
                                val=np.arange(n * m, 2 * n * m).reshape(
                                    (n, m)))

        # Output the elementwise sum of the axiswise sum of matrices M1 ad M2 along the columns
        self.register_output('multiple_matrix_sum_along_1',
                             ot.sum(M1, M2, axes=(1, )))


prob = Problem()
prob.model = ExampleMultipleMatrixAlong1()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('M1', prob['M1'].shape)
print(prob['M1'])
print('M2', prob['M2'].shape)
print(prob['M2'])
print('multiple_matrix_sum_along_1', prob['multiple_matrix_sum_along_1'].shape)
print(prob['multiple_matrix_sum_along_1'])
