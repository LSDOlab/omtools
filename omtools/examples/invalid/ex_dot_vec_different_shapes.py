from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ErrorVecDifferentShapes(Group):
    def setup(self):
        m = 3
        n = 4

        # Shape of the vectors
        vec_shape = (m, )

        # Values for the two vectors
        vec1 = np.arange(m)
        vec2 = np.arange(n, 2 * n)

        # Adding the vectors and tensors to omtools
        vec1 = self.declare_input('vec1', val=vec1)
        vec2 = self.declare_input('vec2', val=vec2)

        # Vector-Vector Dot Product
        self.register_output('VecVecDot', ot.dot(vec1, vec2))


prob = Problem()
prob.model = ErrorVecDifferentShapes()
prob.setup(force_alloc_complex=True)
prob.run_model()
