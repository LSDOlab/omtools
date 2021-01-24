from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleOuterTensorVectorSparse(Group):
    def setup(self):

        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_input('c', val=c)

        self.register_output(
            'einsum_outer2_sparse_derivs',
            ot.einsum(tens,
                      vec,
                      subscripts='hij,k->hijk',
                      partial_format='sparse'))


prob = Problem()
prob.model = ExampleOuterTensorVectorSparse()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('a', prob['a'].shape)
print(prob['a'])
print('c', prob['c'].shape)
print(prob['c'])
print('einsum_outer2_sparse_derivs', prob['einsum_outer2_sparse_derivs'].shape)
print(prob['einsum_outer2_sparse_derivs'])
