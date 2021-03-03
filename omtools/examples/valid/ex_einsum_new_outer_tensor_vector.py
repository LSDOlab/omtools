from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleOuterTensorVector(Group):
    def setup(self):
        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_input('c', val=c)

        # Outer Product of a tensor and a vector
        self.register_output(
            'einsum_outer2',
            ot.einsum_new_api(
                tens,
                vec,
                operation=[(0, 1, 30), (2, ), (0, 1, 30, 2)],
            ))


prob = Problem()
prob.model = ExampleOuterTensorVector()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('a', prob['a'].shape)
print(prob['a'])
print('c', prob['c'].shape)
print(prob['c'])
print('einsum_outer2', prob['einsum_outer2'].shape)
print(prob['einsum_outer2'])
