from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleMultipleVectorSum(Group):
    def setup(self):

        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        # Special operation: sum all the entries of the first and second
        # vector to a single scalar
        self.register_output('einsum_special2',
                             ot.einsum(vec, vec, subscripts='i,j->'))


prob = Problem()
prob.model = ExampleMultipleVectorSum()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('a', prob['a'].shape)
print(prob['a'])
print('einsum_special2', prob['einsum_special2'].shape)
print(prob['einsum_special2'])
