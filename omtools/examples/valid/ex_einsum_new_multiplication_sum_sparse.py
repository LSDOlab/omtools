from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleMultiplicationSumSparse(Group):
    def setup(self):

        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        self.register_output(
            'einsum_special1_sparse_derivs',
            ot.einsum_new_api(vec,
                              vec,
                              operation=[(1, ), (2, ), (2, )],
                              partial_format='sparse'))


prob = Problem()
prob.model = ExampleMultiplicationSumSparse()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('a', prob['a'].shape)
print(prob['a'])
print('einsum_special1_sparse_derivs', prob['einsum_special1_sparse_derivs'].shape)
print(prob['einsum_special1_sparse_derivs'])
