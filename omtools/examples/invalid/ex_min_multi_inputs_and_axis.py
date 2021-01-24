from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ErrorMultiInputsAndAxis(Group):
    def setup(self):
        # Creating the values for two tensors
        val1 = np.array([[1, 5, -8], [10, -3, -5]])
        val2 = np.array([[2, 6, 9], [-1, 2, 4]])

        # Declaring the two input tensors
        tensor1 = self.declare_input('tensor1', val=val1)
        tensor2 = self.declare_input('tensor2', val=val2)

        # Creating the output for matrix multiplication
        self.register_output('ElementwiseMinWithAxis',
                             ot.min(tensor1, tensor2, axis=0))


prob = Problem()
prob.model = ErrorMultiInputsAndAxis()
prob.setup(force_alloc_complex=True)
prob.run_model()
