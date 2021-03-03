from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleVectorVector(Group):
    def setup(self):

        m = 3

        # Shape of the vectors
        vec_shape = (m, )

        # Values for the two vectors
        vec1 = np.arange(m)
        vec2 = np.arange(m, 2 * m)

        # Adding the vectors to omtools
        vec1 = self.declare_input('vec1', val=vec1)
        vec2 = self.declare_input('vec2', val=vec2)

        # Vector-Vector Dot Product
        self.register_output('VecVecDot', ot.dot(vec1, vec2))


prob = Problem()
prob.model = ExampleVectorVector()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('vec1', prob['vec1'].shape)
print(prob['vec1'])
print('vec2', prob['vec2'].shape)
print(prob['vec2'])
print('VecVecDot', prob['VecVecDot'].shape)
print(prob['VecVecDot'])
