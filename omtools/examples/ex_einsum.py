import numpy as np
from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot

# Note: Expansion is not possible with einsum
class Example(Group):
    def setup(self):
        shape1 = (4,)
        shape2 = (5, 4)
        shape3 = (2, 4, 3)

        a = np.arange(4)
        b = np.arange(20).reshape(shape2)
        c = np.arange(24).reshape(shape3)
        
        vec = self.declare_input('a', val=a)
        mat = self.declare_input('b', val=b)
        tens = self.declare_input('c', val=c)

        # Inner Product of 2 vectors
        self.register_output('einsum_inner1', ot.einsum(vec, vec , subscripts = 'i,i->'))

        # Inner Product of a tensor and a vector
        self.register_output('einsum_inner2', ot.einsum(tens, vec , subscripts = 'ijk,j->ik'))

        # Outer Product of 2 vectors
        self.register_output('einsum_outer1', ot.einsum(vec, vec , subscripts = 'i,j->ij'))

        # Outer Product of a tensor and a vector
        self.register_output('einsum_outer2', ot.einsum(tens, vec , subscripts = 'hij,k->hijk'))

        # Transpose of a matrix
        self.register_output('einsum_reorder1', ot.einsum(mat, subscripts = 'ij->ji'))

        # Transpose of a tensor
        self.register_output('einsum_reorder2', ot.einsum(tens, subscripts = 'ijk->kji'))

        # Summation of all the entries of a vector
        self.register_output('einsum_summ1', ot.einsum(vec, subscripts = 'i->'))

        # Summation of all the entries of a tensor
        self.register_output('einsum_summ2', ot.einsum(tens, subscripts = 'ijk->')) 

        # Special operation: summation of all the entries of first vector and scalar multiply the second vector with the computed sum
        self.register_output('einsum_special1', ot.einsum(vec, vec , subscripts = 'i,j->j'))

        # Special operation: sum all the entries of the first and second vector to a single scalar
        self.register_output('einsum_special2', ot.einsum(vec, vec , subscripts = 'i,j->'))


        # All the above operations done with sparse partials (memory efficient when the partials are sparse and large)
        
        self.register_output('einsum_inner1_sparse_derivs', ot.einsum(vec, vec , subscripts = 'i,i->', partial_format = 'sparse'))

        self.register_output('einsum_inner2_sparse_derivs', ot.einsum(tens, vec , subscripts = 'ijk,j->ik', partial_format = 'sparse'))

        self.register_output('einsum_outer1_sparse_derivs', ot.einsum(vec, vec , subscripts = 'i,j->ij', partial_format = 'sparse'))

        self.register_output('einsum_outer2_sparse_derivs', ot.einsum(tens, vec , subscripts = 'hij,k->hijk', partial_format = 'sparse'))

        self.register_output('einsum_reorder1_sparse_derivs', ot.einsum(mat, subscripts = 'ij->ji', partial_format = 'sparse'))

        self.register_output('einsum_reorder2_sparse_derivs', ot.einsum(tens, subscripts = 'ijk->kji', partial_format = 'sparse'))

        self.register_output('einsum_summ1_sparse_derivs', ot.einsum(vec, subscripts = 'i->', partial_format = 'sparse'))

        self.register_output('einsum_summ2_sparse_derivs', ot.einsum(tens, subscripts = 'ijk->', partial_format = 'sparse'))

        self.register_output('einsum_special1_sparse_derivs', ot.einsum(vec, vec , subscripts = 'i,j->j', partial_format = 'sparse'))

        self.register_output('einsum_special2_sparse_derivs', ot.einsum(vec, vec , subscripts = 'i,j->', partial_format = 'sparse'))

prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.check_partials(compact_print=True)
prob.run_model()
