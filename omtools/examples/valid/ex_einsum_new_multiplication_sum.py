from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleMultiplicationSum(Group):
    def setup(self):

        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        # Special operation: summation of all the entries of first
        # vector and scalar multiply the second vector with the computed
        # sum
        self.register_output(
            'einsum_special1',
            ot.einsum_new_api(
                vec,
                vec,
                operation=[(1, ), (2, ), (2, )],
            ))


prob = Problem()
prob.model = ExampleMultiplicationSum()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('a', prob['a'].shape)
print(prob['a'])
print('einsum_special1', prob['einsum_special1'].shape)
print(prob['einsum_special1'])
