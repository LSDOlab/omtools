from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleInnerTensorVector(Group):
    def setup(self):
        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_input('c', val=c)

        # Inner Product of a tensor and a vector
        self.register_output('einsum_inner2',
                             ot.einsum(
                                 tens,
                                 vec,
                                 subscripts='ijk,j->ik',
                             ))


prob = Problem()
prob.model = ExampleInnerTensorVector()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('a', prob['a'].shape)
print(prob['a'])
print('c', prob['c'].shape)
print(prob['c'])
print('einsum_inner2', prob['einsum_inner2'].shape)
print(prob['einsum_inner2'])
