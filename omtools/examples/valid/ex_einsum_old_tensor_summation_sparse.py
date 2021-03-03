from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleTensorSummationSparse(Group):
    def setup(self):
        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_input('c', val=c)

        self.register_output(
            'einsum_summ2_sparse_derivs',
            ot.einsum(
                tens,
                subscripts='ijk->',
                partial_format='sparse',
            ))


prob = Problem()
prob.model = ExampleTensorSummationSparse()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('c', prob['c'].shape)
print(prob['c'])
print('einsum_summ2_sparse_derivs', prob['einsum_summ2_sparse_derivs'].shape)
print(prob['einsum_summ2_sparse_derivs'])
