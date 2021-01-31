from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleSingleMatrixAlong0(Group):
    def setup(self):
        n = 3
        m = 6

        # Declare a matrix of shape 3x6 as input
        M1 = self.declare_input('M1', val=np.arange(n * m).reshape((n, m)))

        # Output the axiswise average of matrix M1 along the columns
        self.register_output('single_matrix_average_along_0',
                             ot.average(M1, axes=(0, )))


prob = Problem()
prob.model = ExampleSingleMatrixAlong0()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('M1', prob['M1'].shape)
print(prob['M1'])
print('single_matrix_average_along_0', prob['single_matrix_average_along_0'].shape)
print(prob['single_matrix_average_along_0'])
