import numpy as np
from omtools.api import Group
import omtools.api as ot


# Note: Expansion is not possible with einsum
class ExampleInnerVectorVector(Group):
    """
    :param var: a
    :param var: einsum_inner1
    """
    def setup(self):
        a = np.arange(4)

        vec = self.declare_input('a', val=a)

        # Inner Product of 2 vectors
        self.register_output(
            'einsum_inner1',
            ot.einsum_new_api(vec, vec, operation=[(0, ), (0, )]))


class ExampleInnerTensorVector(Group):
    """
    :param var: a
    :param var: c
    :param var: einsum_inner2
    """
    def setup(self):
        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_input('c', val=c)

        # Inner Product of a tensor and a vector
        self.register_output(
            'einsum_inner2',
            ot.einsum_new_api(
                tens,
                vec,
                operation=[('rows', 0, 1), (0, ), ('rows', 1)],
            ))


class ExampleOuterVectorVector(Group):
    """
    :param var: a
    :param var: einsum_outer1
    """
    def setup(self):
        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        self.register_output(
            'einsum_outer1',
            ot.einsum_new_api(
                vec,
                vec,
                operation=[('rows', ), ('cols', ), ('rows', 'cols')],
            ))


class ExampleOuterTensorVector(Group):
    """
    :param var: a
    :param var: c
    :param var: einsum_outer2
    """
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


class ExampleReorderMatrix(Group):
    """
    :param var: b
    :param var: einsum_reorder1
    """
    def setup(self):
        shape2 = (5, 4)
        b = np.arange(20).reshape(shape2)
        mat = self.declare_input('b', val=b)

        # reorder of a matrix
        self.register_output(
            'einsum_reorder1',
            ot.einsum_new_api(mat, operation=[(46, 99), (99, 46)]))


class ExampleReorderTensor(Group):
    """
    :param var: c
    :param var: einsum_reorder2
    """
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


class ExampleVectorSummation(Group):
    """
    :param var: a
    :param var: einsum_summ1
    """
    def setup(self):
        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        # Summation of all the entries of a vector
        self.register_output('einsum_summ1',
                             ot.einsum_new_api(vec, operation=[(33, )]))


class ExampleTensorSummation(Group):
    """
    :param var: c
    :param var: einsum_summ2
    """
    def setup(self):
        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_input('c', val=c)

        # Summation of all the entries of a tensor
        self.register_output(
            'einsum_summ2', ot.einsum_new_api(
                tens,
                operation=[(33, 66, 99)],
            ))


class ExampleMultiplicationSum(Group):
    """
    :param var: a
    :param var: einsum_special1
    """
    def setup(self):

        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        # Special operation: summation of all the entries of first
        # vector and scalar multiply the second vector with the computed
        # sum
        self.register_output(
            'einsum_special1',
            ot.einsum_new_api(
                vec,
                vec,
                operation=[(1, ), (2, ), (2, )],
            ))


class ExampleMultipleVectorSum(Group):
    """
    :param var: a
    :param var: einsum_special2
    """
    def setup(self):

        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        # Special operation: sum all the entries of the first and second
        # vector to a single scalar
        self.register_output(
            'einsum_special2',
            ot.einsum_new_api(vec, vec, operation=[(1, ), (2, )]))


# All the above operations done with sparse partials (memory
# efficient when the partials are sparse and large)


class ExampleInnerVectorVectorSparse(Group):
    """
    :param var: a
    :param var: einsum_inner1_sparse_derivs
    """
    def setup(self):
        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        self.register_output(
            'einsum_inner1_sparse_derivs',
            ot.einsum_new_api(vec,
                              vec,
                              operation=[(0, ), (0, )],
                              partial_format='sparse'))


class ExampleInnerTensorVectorSparse(Group):
    """
    :param var: a
    :param var: c
    :param var: einsum_inner2_sparse_derivs
    """
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


class ExampleOuterVectorVectorSparse(Group):
    """
    :param var: a
    :param var: einsum_outer1_sparse_derivs
    """
    def setup(self):

        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        self.register_output(
            'einsum_outer1_sparse_derivs',
            ot.einsum_new_api(vec,
                              vec,
                              operation=[('rows', ), ('cols', ),
                                         ('rows', 'cols')],
                              partial_format='sparse'))


class ExampleOuterTensorVectorSparse(Group):
    """
    :param var: a
    :param var: c
    :param var: einsum_outer2_sparse_derivs
    """
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
            ot.einsum_new_api(tens,
                              vec,
                              operation=[
                                  (0, 1, 30),
                                  (2, ),
                                  (0, 1, 30, 2),
                              ],
                              partial_format='sparse'))


class ExampleReorderMatrixSparse(Group):
    """
    :param var: b
    :param var: einsum_reorder1_sparse_derivs
    """
    def setup(self):

        shape2 = (5, 4)
        b = np.arange(20).reshape(shape2)
        mat = self.declare_input('b', val=b)

        self.register_output(
            'einsum_reorder1_sparse_derivs',
            ot.einsum_new_api(mat,
                              operation=[(46, 99), (99, 46)],
                              partial_format='sparse'))


class ExampleReorderTensorSparse(Group):
    """
    :param var: c
    :param var: einsum_reorder2_sparse_derivs
    """
    def setup(self):

        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_input('c', val=c)

        self.register_output(
            'einsum_reorder2_sparse_derivs',
            ot.einsum_new_api(tens,
                              operation=[(33, 66, 99), (99, 66, 33)],
                              partial_format='sparse'))


class ExampleVectorSummationSparse(Group):
    """
    :param var: a
    :param var: einsum_summ1_sparse_derivs
    """
    def setup(self):
        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        self.register_output(
            'einsum_summ1_sparse_derivs',
            ot.einsum_new_api(vec, operation=[(33, )],
                              partial_format='sparse'))


class ExampleTensorSummationSparse(Group):
    """
    :param var: c
    :param var: einsum_summ2_sparse_derivs
    """
    def setup(self):
        # Shape of Tensor
        shape3 = (2, 4, 3)
        c = np.arange(24).reshape(shape3)

        # Declaring tensor
        tens = self.declare_input('c', val=c)

        self.register_output(
            'einsum_summ2_sparse_derivs',
            ot.einsum_new_api(tens,
                              operation=[(33, 66, 99)],
                              partial_format='sparse'))


class ExampleMultiplicationSumSparse(Group):
    """
    :param var: a
    :param var: einsum_special1_sparse_derivs
    """
    def setup(self):

        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        self.register_output(
            'einsum_special1_sparse_derivs',
            ot.einsum_new_api(vec,
                              vec,
                              operation=[(1, ), (2, ), (2, )],
                              partial_format='sparse'))


class ExampleMultipleVectorSumSparse(Group):
    """
    :param var: a
    :param var: einsum_special2_sparse_derivs
    """
    def setup(self):

        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        self.register_output(
            'einsum_special2_sparse_derivs',
            ot.einsum_new_api(vec,
                              vec,
                              operation=[
                                  (1, ),
                                  (2, ),
                              ],
                              partial_format='sparse'))
