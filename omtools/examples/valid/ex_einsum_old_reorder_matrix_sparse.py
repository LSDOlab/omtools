from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleReorderMatrixSparse(Group):
    def setup(self):

        shape2 = (5, 4)
        b = np.arange(20).reshape(shape2)
        mat = self.declare_input('b', val=b)

        self.register_output(
            'einsum_reorder1_sparse_derivs',
            ot.einsum(
                mat,
                subscripts='ij->ji',
                partial_format='sparse',
            ))


prob = Problem()
prob.model = ExampleReorderMatrixSparse()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('b', prob['b'].shape)
print(prob['b'])
print('einsum_reorder1_sparse_derivs', prob['einsum_reorder1_sparse_derivs'].shape)
print(prob['einsum_reorder1_sparse_derivs'])
