from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleMatrix(Group):
    """
    :param var: Mat
    :param var: matrix_transpose
    """
    def setup(self):

        # Declare mat as an input matrix with shape = (4, 2)
        mat = self.declare_input(
            'Mat',
            val=np.arange(4 * 2).reshape((4, 2)),
        )

        # Compute the transpose of mat
        self.register_output('matrix_transpose', ot.transpose(mat))


class ExampleTensor(Group):
    """
    :param var: Tens
    :param var: tensor_transpose
    """
    def setup(self):

        # Declare tens as an input tensor with shape = (4, 3, 2, 5)
        tens = self.declare_input(
            'Tens',
            val=np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2)),
        )

        # Compute the transpose of tens
        self.register_output('tensor_transpose', ot.transpose(tens))
