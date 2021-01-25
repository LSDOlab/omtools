from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleReorderTensor(Group):
    def setup(self):

        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_input('c', val=c)

        # reorder of a tensor
        self.register_output(
            'einsum_reorder2',
            ot.einsum_new_api(
                tens,
                operation=[(33, 66, 99), (99, 66, 33)],
            ))


prob = Problem()
prob.model = ExampleReorderTensor()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('c', prob['c'].shape)
print(prob['c'])
print('einsum_reorder2', prob['einsum_reorder2'].shape)
print(prob['einsum_reorder2'])
