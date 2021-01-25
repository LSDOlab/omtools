from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleMatrix(Group):
    """
    :param var: M1
    :param var: axes_reordered_matrix
    """
    def setup(self):

        # Declare mat as an input matrix with shape = (4, 2)
        mat = self.declare_input(
            'M1',
            val=np.arange(4 * 2).reshape((4, 2)),
        )

        # Compute the transpose of mat
        self.register_output('axes_reordered_matrix',
                             ot.reorder_axes(mat, 'ij->ji'))


class ExampleTensor(Group):
    """
    :param var: T1
    :param var: axes_reordered_tensor
    """
    def setup(self):

        # Declare tens as an input tensor with shape = (4, 3, 2, 5)
        tens = self.declare_input(
            'T1',
            val=np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2)),
        )

        # Compute a new tensor by reordering axes of tens; reordering is
        # given by 'ijkl->ljki'
        self.register_output('axes_reordered_tensor',
                             ot.reorder_axes(tens, 'ijkl->ljki'))