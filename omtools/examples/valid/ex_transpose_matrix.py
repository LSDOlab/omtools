from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleMatrix(Group):
    def setup(self):

        # Declare mat as an input matrix with shape = (4, 2)
        mat = self.declare_input(
            'Mat',
            val=np.arange(4 * 2).reshape((4, 2)),
        )

        # Compute the transpose of mat
        self.register_output('matrix_transpose', ot.transpose(mat))


prob = Problem()
prob.model = ExampleMatrix()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('Mat', prob['Mat'].shape)
print(prob['Mat'])
print('matrix_transpose', prob['matrix_transpose'].shape)
print(prob['matrix_transpose'])
