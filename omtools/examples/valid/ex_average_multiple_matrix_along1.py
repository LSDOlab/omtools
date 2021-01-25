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

        # Output the elementwise average of the axiswise average of matrices M1 ad M2 along the columns
        self.register_output('multiple_matrix_average_along_1',
                             ot.average(M1, M2, axes=(1, )))


prob = Problem()
prob.model = ExampleMultipleMatrixAlong1()
prob.setup(force_alloc_complex=True)
prob.run_model()
