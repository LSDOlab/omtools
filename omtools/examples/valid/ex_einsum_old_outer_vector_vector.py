from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleOuterVectorVector(Group):
    def setup(self):
        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        # Outer Product of 2 vectors
        self.register_output('einsum_outer1',
                             ot.einsum(vec, vec, subscripts='i,j->ij'))


prob = Problem()
prob.model = ExampleOuterVectorVector()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('a', prob['a'].shape)
print(prob['a'])
print('einsum_outer1', prob['einsum_outer1'].shape)
print(prob['einsum_outer1'])
