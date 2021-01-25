from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleInnerTensorVectorSparse(Group):
    def setup(self):

        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_input('c', val=c)

        self.register_output(
            'einsum_inner2_sparse_derivs',
            ot.einsum_new_api(tens,
                              vec,
                              operation=[
                                  ('rows', 0, 1),
                                  (0, ),
                                  ('rows', 1),
                              ],
                              partial_format='sparse'))


prob = Problem()
prob.model = ExampleInnerTensorVectorSparse()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('a', prob['a'].shape)
print(prob['a'])
print('c', prob['c'].shape)
print(prob['c'])
print('einsum_inner2_sparse_derivs', prob['einsum_inner2_sparse_derivs'].shape)
print(prob['einsum_inner2_sparse_derivs'])
