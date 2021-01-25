from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleInnerVectorVector(Group):
    def setup(self):
        a = np.arange(4)

        vec = self.declare_input('a', val=a)

        # Inner Product of 2 vectors
        self.register_output('einsum_inner1',
                             ot.einsum(vec, vec, subscripts='i,i->'))


prob = Problem()
prob.model = ExampleInnerVectorVector()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('a', prob['a'].shape)
print(prob['a'])
print('einsum_inner1', prob['einsum_inner1'].shape)
print(prob['einsum_inner1'])
