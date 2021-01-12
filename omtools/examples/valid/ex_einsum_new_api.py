from openmdao.api import Problem
import numpy as np
from omtools.api import Group
import omtools.api as ot


class ExampleNewAPI(Group):
    def setup(self):
        shape1 = (4, )
        shape2 = (5, 4)
        shape3 = (2, 4, 3)

        a = np.arange(4)
        b = np.arange(20).reshape(shape2)
        c = np.arange(24).reshape(shape3)

        vec = self.declare_input('a', val=a)
        mat = self.declare_input('b', val=b)
        tens = self.declare_input('c', val=c)

        # Inner Product of 2 vectors
        self.register_output(
            'new_einsum_inner1',
            ot.einsum_new_api(vec, vec, operation=[(0, ), (0, )]))

        # Inner Product of a tensor and a vector
        self.register_output(
            'new_einsum_inner2',
            ot.einsum_new_api(
                tens,
                vec,
                operation=[('rows', 0, 1), (0, ), ('rows', 1)],
            ))

        # Outer Product of 2 vectors
        self.register_output(
            'new_einsum_outer1',
            ot.einsum_new_api(
                vec,
                vec,
                operation=[('rows', ), ('cols', ), ('rows', 'cols')],
            ))

        # Outer Product of a tensor and a vector
        self.register_output(
            'new_einsum_outer2',
            ot.einsum_new_api(
                tens,
                vec,
                operation=[(0, 1, 30), (2, ), (0, 1, 30, 2)],
            ))

        # Transpose of a matrix
        self.register_output(
            'new_einsum_reorder1',
            ot.einsum_new_api(mat, operation=[(46, 99), (99, 46)]))

        # Transpose of a tensor
        self.register_output(
            'new_einsum_reorder2',
            ot.einsum_new_api(
                tens,
                operation=[(33, 66, 99), (99, 66, 33)],
            ))

        # Summation of all the entries of a vector
        self.register_output('new_einsum_summ1',
                             ot.einsum_new_api(vec, operation=[(33, )]))

        # Summation of all the entries of a tensor
        self.register_output(
            'new_einsum_summ2',
            ot.einsum_new_api(
                tens,
                operation=[(33, 66, 99)],
            ))

        # Special operation: summation of all the entries of the first
        # vector and scalar multiply the second vector with the computed
        # sum
        self.register_output(
            'new_einsum_special1',
            ot.einsum_new_api(
                vec,
                vec,
                operation=[(1, ), (2, ), (2, )],
            ))

        # Special operation: sum all the entries of the first and second
        # vector to a single scalar
        self.register_output(
            'new_einsum_special2',
            ot.einsum_new_api(vec, vec, operation=[(1, ), (2, )]))

        # All the above operations done with sparse partials (memory
        # efficient when the partials are sparse and large)

        self.register_output(
            'new_einsum_inner1_sparse_derivs',
            ot.einsum_new_api(vec,
                              vec,
                              operation=[(0, ), (0, )],
                              partial_format='sparse'))

        self.register_output(
            'new_einsum_inner2_sparse_derivs',
            ot.einsum_new_api(tens,
                              vec,
                              operation=[
                                  ('rows', 0, 1),
                                  (0, ),
                                  ('rows', 1),
                              ],
                              partial_format='sparse'))
        self.register_output(
            'new_einsum_outer1_sparse_derivs',
            ot.einsum_new_api(vec,
                              vec,
                              operation=[('rows', ), ('cols', ),
                                         ('rows', 'cols')],
                              partial_format='sparse'))

        self.register_output(
            'new_einsum_outer2_sparse_derivs',
            ot.einsum_new_api(tens,
                              vec,
                              operation=[
                                  (0, 1, 30),
                                  (2, ),
                                  (0, 1, 30, 2),
                              ],
                              partial_format='sparse'))
        self.register_output(
            'new_einsum_reorder1_sparse_derivs',
            ot.einsum_new_api(mat,
                              operation=[(46, 99), (99, 46)],
                              partial_format='sparse'))

        self.register_output(
            'new_einsum_reorder2_sparse_derivs',
            ot.einsum_new_api(tens,
                              operation=[(33, 66, 99), (99, 66, 33)],
                              partial_format='sparse'))

        self.register_output(
            'new_einsum_summ1_sparse_derivs',
            ot.einsum_new_api(vec, operation=[(33, )],
                              partial_format='sparse'))

        self.register_output(
            'new_einsum_summ2_sparse_derivs',
            ot.einsum_new_api(tens,
                              operation=[(33, 66, 99)],
                              partial_format='sparse'))

        self.register_output(
            'new_einsum_special1_sparse_derivs',
            ot.einsum_new_api(vec,
                              vec,
                              operation=[(1, ), (2, ), (2, )],
                              partial_format='sparse'))

        self.register_output(
            'new_einsum_special2_sparse_derivs',
            ot.einsum_new_api(vec,
                              vec,
                              operation=[
                                  (1, ),
                                  (2, ),
                              ],
                              partial_format='sparse'))


prob = Problem()
prob.model = ExampleNewAPI()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('new_einsum_inner1', prob['new_einsum_inner1'].shape)
print(prob['new_einsum_inner1'])
print('new_einsum_inner2', prob['new_einsum_inner2'].shape)
print(prob['new_einsum_inner2'])
print('new_einsum_outer1', prob['new_einsum_outer1'].shape)
print(prob['new_einsum_outer1'])
print('new_einsum_outer2', prob['new_einsum_outer2'].shape)
print(prob['new_einsum_outer2'])
print('new_einsum_reorder1', prob['new_einsum_reorder1'].shape)
print(prob['new_einsum_reorder1'])
print('new_einsum_reorder2', prob['new_einsum_reorder2'].shape)
print(prob['new_einsum_reorder2'])
print('new_einsum_summ1', prob['new_einsum_summ1'].shape)
print(prob['new_einsum_summ1'])
print('new_einsum_summ2', prob['new_einsum_summ2'].shape)
print(prob['new_einsum_summ2'])
print('new_einsum_special1', prob['new_einsum_special1'].shape)
print(prob['new_einsum_special1'])
print('new_einsum_special2', prob['new_einsum_special2'].shape)
print(prob['new_einsum_special2'])
print('new_einsum_inner1_sparse_derivs', prob['new_einsum_inner1_sparse_derivs'].shape)
print(prob['new_einsum_inner1_sparse_derivs'])
print('new_einsum_inner2_sparse_derivs', prob['new_einsum_inner2_sparse_derivs'].shape)
print(prob['new_einsum_inner2_sparse_derivs'])
print('new_einsum_outer1_sparse_derivs', prob['new_einsum_outer1_sparse_derivs'].shape)
print(prob['new_einsum_outer1_sparse_derivs'])
print('new_einsum_outer2_sparse_derivs', prob['new_einsum_outer2_sparse_derivs'].shape)
print(prob['new_einsum_outer2_sparse_derivs'])
print('new_einsum_reorder1_sparse_derivs', prob['new_einsum_reorder1_sparse_derivs'].shape)
print(prob['new_einsum_reorder1_sparse_derivs'])
print('new_einsum_reorder2_sparse_derivs', prob['new_einsum_reorder2_sparse_derivs'].shape)
print(prob['new_einsum_reorder2_sparse_derivs'])
print('new_einsum_summ1_sparse_derivs', prob['new_einsum_summ1_sparse_derivs'].shape)
print(prob['new_einsum_summ1_sparse_derivs'])
print('new_einsum_summ2_sparse_derivs', prob['new_einsum_summ2_sparse_derivs'].shape)
print(prob['new_einsum_summ2_sparse_derivs'])
print('new_einsum_special1_sparse_derivs', prob['new_einsum_special1_sparse_derivs'].shape)
print(prob['new_einsum_special1_sparse_derivs'])
print('new_einsum_special2_sparse_derivs', prob['new_einsum_special2_sparse_derivs'].shape)
print(prob['new_einsum_special2_sparse_derivs'])
