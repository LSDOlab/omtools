from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleVectorSummation(Group):
    def setup(self):
        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        # Summation of all the entries of a vector
        self.register_output('einsum_summ1',
                             ot.einsum_new_api(vec, operation=[(33, )]))


prob = Problem()
prob.model = ExampleVectorSummation()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('a', prob['a'].shape)
print(prob['a'])
print('einsum_summ1', prob['einsum_summ1'].shape)
print(prob['einsum_summ1'])
