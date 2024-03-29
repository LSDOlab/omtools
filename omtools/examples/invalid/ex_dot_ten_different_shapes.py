from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ErrorTenDifferentShapes(Group):
    def setup(self):
        m = 3
        n = 4
        p = 5

        # Shape of the tensors
        ten_shape1 = (m, n, p)
        ten_shape2 = (n, n, m)

        # Number of elements in the tensors
        num_ten_elements1 = np.prod(ten_shape1)
        num_ten_elements2 = np.prod(ten_shape2)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements1).reshape(ten_shape1)
        ten2 = np.arange(num_ten_elements2,
                         2 * num_ten_elements2).reshape(ten_shape2)

        ten1 = self.declare_input('ten1', val=ten1)
        ten2 = self.declare_input('ten2', val=ten2)

        # Tensor-Tensor Dot Product specifying the first axis
        self.register_output('TenTenDotFirst', ot.dot(ten1, ten2, axis=0))


prob = Problem()
prob.model = ErrorTenDifferentShapes()
prob.setup(force_alloc_complex=True)
prob.run_model()
