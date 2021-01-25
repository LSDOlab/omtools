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
        self.register_output('einsum_inner1',
                             ot.einsum(vec, vec, subscripts='i,i->'))


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
        self.register_output('einsum_inner2',
                             ot.einsum(
                                 tens,
                                 vec,
                                 subscripts='ijk,j->ik',
                             ))


class ExampleOuterVectorVector(Group):
    """
    :param var: a
    :param var: einsum_outer1
    """
    def setup(self):
        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        # Outer Product of 2 vectors
        self.register_output('einsum_outer1',
                             ot.einsum(vec, vec, subscripts='i,j->ij'))


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
        self.register_output('einsum_outer2',
                             ot.einsum(
                                 tens,
                                 vec,
                                 subscripts='hij,k->hijk',
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

        # Transpose of a matrix
        self.register_output('einsum_reorder1',
                             ot.einsum(mat, subscripts='ij->ji'))


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

        # Transpose of a tensor
        self.register_output('einsum_reorder2',
                             ot.einsum(tens, subscripts='ijk->kji'))


class ExampleVectorSummation(Group):
    """
    :param var: a
    :param var: einsum_summ1
    """
    def setup(self):
        a = np.arange(4)
        vec = self.declare_input('a', val=a)

        # Summation of all the entries of a vector
        self.register_output('einsum_summ1', ot.einsum(
            vec,
            subscripts='i->',
        ))


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
        self.register_output('einsum_summ2',
                             ot.einsum(
                                 tens,
                                 subscripts='ijk->',
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
        self.register_output('einsum_special1',
                             ot.einsum(vec, vec, subscripts='i,j->j'))


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
        self.register_output('einsum_special2',
                             ot.einsum(vec, vec, subscripts='i,j->'))


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
            ot.einsum(
                vec,
                vec,
                subscripts='i,i->',
                partial_format='sparse',
            ))


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
            ot.einsum(tens,
                      vec,
                      subscripts='ijk,j->ik',
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
            ot.einsum(
                vec,
                vec,
                subscripts='i,j->ij',
                partial_format='sparse',
            ))


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
            ot.einsum(tens,
                      vec,
                      subscripts='hij,k->hijk',
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
            ot.einsum(
                mat,
                subscripts='ij->ji',
                partial_format='sparse',
            ))


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
            ot.einsum(
                tens,
                subscripts='ijk->kji',
                partial_format='sparse',
            ))


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
            ot.einsum(vec, subscripts='i->', partial_format='sparse'))


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
            ot.einsum(
                tens,
                subscripts='ijk->',
                partial_format='sparse',
            ))


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
            ot.einsum(
                vec,
                vec,
                subscripts='i,j->j',
                partial_format='sparse',
            ))


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
            ot.einsum(
                vec,
                vec,
                subscripts='i,j->',
                partial_format='sparse',
            ))
