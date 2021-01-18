import numpy as np
from omtools.api import Group
import omtools.api as ot


# Note: Expansion is not possible with einsum
class ExampleNewAPI(Group):
    """
    :param var: new_einsum_inner1
    :param var: new_einsum_inner2
    :param var: new_einsum_outer1
    :param var: new_einsum_outer2
    :param var: new_einsum_reorder1
    :param var: new_einsum_reorder2
    :param var: new_einsum_summ1
    :param var: new_einsum_summ2
    :param var: new_einsum_special1
    :param var: new_einsum_special2
    :param var: new_einsum_inner1_sparse_derivs
    :param var: new_einsum_inner2_sparse_derivs
    :param var: new_einsum_outer1_sparse_derivs
    :param var: new_einsum_outer2_sparse_derivs
    :param var: new_einsum_reorder1_sparse_derivs
    :param var: new_einsum_reorder2_sparse_derivs
    :param var: new_einsum_summ1_sparse_derivs
    :param var: new_einsum_summ2_sparse_derivs
    :param var: new_einsum_special1_sparse_derivs
    :param var: new_einsum_special2_sparse_derivs
    """
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


# Note: Expansion is not possible with einsum
class ExampleOldAPI(Group):
    """
    :param var: einsum_inner1
    :param var: einsum_inner2
    :param var: einsum_outer1
    :param var: einsum_outer2
    :param var: einsum_reorder1
    :param var: einsum_reorder2
    :param var: einsum_summ1
    :param var: einsum_summ2
    :param var: einsum_special1
    :param var: einsum_special2
    :param var: einsum_inner1_sparse_derivs
    :param var: einsum_inner2_sparse_derivs
    :param var: einsum_outer1_sparse_derivs
    :param var: einsum_outer2_sparse_derivs
    :param var: einsum_reorder1_sparse_derivs
    :param var: einsum_reorder2_sparse_derivs
    :param var: einsum_summ1_sparse_derivs
    :param var: einsum_summ2_sparse_derivs
    :param var: einsum_special1_sparse_derivs
    :param var: einsum_special2_sparse_derivs
    """
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
        self.register_output('einsum_inner1',
                             ot.einsum(vec, vec, subscripts='i,i->'))

        # Inner Product of a tensor and a vector
        self.register_output('einsum_inner2',
                             ot.einsum(
                                 tens,
                                 vec,
                                 subscripts='ijk,j->ik',
                             ))

        # Outer Product of 2 vectors
        self.register_output('einsum_outer1',
                             ot.einsum(vec, vec, subscripts='i,j->ij'))

        # Outer Product of a tensor and a vector
        self.register_output('einsum_outer2',
                             ot.einsum(
                                 tens,
                                 vec,
                                 subscripts='hij,k->hijk',
                             ))

        # Transpose of a matrix
        self.register_output('einsum_reorder1',
                             ot.einsum(mat, subscripts='ij->ji'))

        # Transpose of a tensor
        self.register_output('einsum_reorder2',
                             ot.einsum(tens, subscripts='ijk->kji'))

        # Summation of all the entries of a vector
        self.register_output('einsum_summ1', ot.einsum(
            vec,
            subscripts='i->',
        ))

        # Summation of all the entries of a tensor
        self.register_output('einsum_summ2',
                             ot.einsum(
                                 tens,
                                 subscripts='ijk->',
                             ))

        # Special operation: summation of all the entries of first
        # vector and scalar multiply the second vector with the computed
        # sum
        self.register_output('einsum_special1',
                             ot.einsum(vec, vec, subscripts='i,j->j'))

        # Special operation: sum all the entries of the first and second
        # vector to a single scalar
        self.register_output('einsum_special2',
                             ot.einsum(vec, vec, subscripts='i,j->'))

        # All the above operations done with sparse partials (memory
        # efficient when the partials are sparse and large)

        self.register_output(
            'einsum_inner1_sparse_derivs',
            ot.einsum(
                vec,
                vec,
                subscripts='i,i->',
                partial_format='sparse',
            ))

        self.register_output(
            'einsum_inner2_sparse_derivs',
            ot.einsum(tens,
                      vec,
                      subscripts='ijk,j->ik',
                      partial_format='sparse'))

        self.register_output(
            'einsum_outer1_sparse_derivs',
            ot.einsum(
                vec,
                vec,
                subscripts='i,j->ij',
                partial_format='sparse',
            ))

        self.register_output(
            'einsum_outer2_sparse_derivs',
            ot.einsum(tens,
                      vec,
                      subscripts='hij,k->hijk',
                      partial_format='sparse'))

        self.register_output(
            'einsum_reorder1_sparse_derivs',
            ot.einsum(
                mat,
                subscripts='ij->ji',
                partial_format='sparse',
            ))

        self.register_output(
            'einsum_reorder2_sparse_derivs',
            ot.einsum(
                tens,
                subscripts='ijk->kji',
                partial_format='sparse',
            ))

        self.register_output(
            'einsum_summ1_sparse_derivs',
            ot.einsum(vec, subscripts='i->', partial_format='sparse'))

        self.register_output(
            'einsum_summ2_sparse_derivs',
            ot.einsum(
                tens,
                subscripts='ijk->',
                partial_format='sparse',
            ))

        self.register_output(
            'einsum_special1_sparse_derivs',
            ot.einsum(
                vec,
                vec,
                subscripts='i,j->j',
                partial_format='sparse',
            ))

        self.register_output(
            'einsum_special2_sparse_derivs',
            ot.einsum(
                vec,
                vec,
                subscripts='i,j->',
                partial_format='sparse',
            ))
