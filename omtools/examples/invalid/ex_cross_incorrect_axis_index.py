from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ErrorIncorrectAxisIndex(Group):
    def setup(self):
        # Creating two tensors
        shape = (2, 5, 4, 3)
        num_elements = np.prod(shape)

        tenval1 = np.arange(num_elements).reshape(shape)
        tenval2 = np.arange(num_elements).reshape(shape) + 6

        ten1 = self.declare_input('ten1', val=tenval1)
        ten2 = self.declare_input('ten2', val=tenval2)

        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('TenTenCross', ot.cross(ten1, ten2, axis=2))


prob = Problem()
prob.model = ErrorIncorrectAxisIndex()
prob.setup(force_alloc_complex=True)
prob.run_model()
