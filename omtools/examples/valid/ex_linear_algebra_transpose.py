from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleTranspose(Group):
    def setup(self):

        # Declare mat as an input matrix with shape = (4, 2)
        mat = self.declare_input(
            'Mat',
            val=np.arange(4 * 2).reshape((4, 2)),
        )

        # Declare tens as an input tensor with shape = (4, 3, 2, 5)
        tens = self.declare_input(
            'Tens',
            val=np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2)),
        )

        # Compute the transpose of mat
        self.register_output('matrix_transpose', ot.transpose(mat))

        # Compute the transpose of tens
        self.register_output('tensor_transpose', ot.transpose(tens))


prob = Problem()
prob.model = ExampleTranspose()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('matrix_transpose', prob['matrix_transpose'].shape)
print(prob['matrix_transpose'])
print('tensor_transpose', prob['tensor_transpose'].shape)
print(prob['tensor_transpose'])
