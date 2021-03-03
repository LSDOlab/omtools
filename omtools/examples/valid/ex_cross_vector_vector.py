from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleVectorVector(Group):
    def setup(self):
        # Creating two vectors
        vecval1 = np.arange(3)
        vecval2 = np.arange(3) + 1

        vec1 = self.declare_input('vec1', val=vecval1)
        vec2 = self.declare_input('vec2', val=vecval2)

        # Vector-Vector Cross Product
        self.register_output('VecVecCross', ot.cross(vec1, vec2, axis=0))


prob = Problem()
prob.model = ExampleVectorVector()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('vec1', prob['vec1'].shape)
print(prob['vec1'])
print('vec2', prob['vec2'].shape)
print(prob['vec2'])
print('VecVecCross', prob['VecVecCross'].shape)
print(prob['VecVecCross'])
