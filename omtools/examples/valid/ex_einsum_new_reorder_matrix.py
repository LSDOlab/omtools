from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleReorderMatrix(Group):
    def setup(self):
        shape2 = (5, 4)
        b = np.arange(20).reshape(shape2)
        mat = self.declare_input('b', val=b)

        # reorder of a matrix
        self.register_output(
            'einsum_reorder1',
            ot.einsum_new_api(mat, operation=[(46, 99), (99, 46)]))


prob = Problem()
prob.model = ExampleReorderMatrix()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('b', prob['b'].shape)
print(prob['b'])
print('einsum_reorder1', prob['einsum_reorder1'].shape)
print(prob['einsum_reorder1'])
