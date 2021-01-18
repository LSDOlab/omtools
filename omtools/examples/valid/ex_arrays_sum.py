from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleSum(Group):
    """
    
    """
    def setup(self):
        n = 3
        m = 6
        p = 7
        q = 10

        # Declare a vector of length 3 as input
        v1 = self.declare_input('v1', val=np.arange(n))

        # Declare another vector of length 3 as input
        v2 = self.declare_input('v2', val=np.arange(n, 2 * n))

        # Declare a matrix of shape 3x6 as input
        M1 = self.declare_input('M1', val=np.arange(n * m).reshape((n, m)))

        # Declare another matrix of shape 3x6 as input
        M2 = self.declare_input('M2',
                                val=np.arange(n * m, 2 * n * m).reshape(
                                    (n, m)))

        # Declare a tensor of shape 3x6x7x10 as input
        T1 = self.declare_input('T1',
                                val=np.arange(n * m * p * q).reshape(
                                    (n, m, p, q)))

        # Declare another tensor of shape 3x6x7x10 as input
        T2 = self.declare_input('T2',
                                val=np.arange(n * m * p * q,
                                              2 * n * m * p * q).reshape(
                                                  (n, m, p, q)))

        # Output the sum of all the elements of the vector v1
        self.register_output('single_vector_sum', ot.sum(v1))

        # Output the sum of all the elements of the matrix M1
        self.register_output('single_matrix_sum', ot.sum(M1))

        # Output the sum of all the elements of the tensor T1
        self.register_output('single_tensor_sum', ot.sum(T1))

        # Output the elementwise sum of vectors v1 and v2
        self.register_output('multiple_vector_sum', ot.sum(v1, v2))

        # Output the elementwise sum of matrices M1 and M2
        self.register_output('multiple_matrix_sum', ot.sum(M1, M2))

        # Output the elementwise sum of tensors T1 and T2
        self.register_output('multiple_tensor_sum', ot.sum(T1, T2))

        # Output the axiswise sum of matrix M1 along the columns
        self.register_output('single_matrix_sum_along_0', ot.sum(M1,
                                                                 axes=(0, )))

        # Output the axiswise sum of matrix M1 along the rows
        self.register_output('single_matrix_sum_along_1', ot.sum(M1,
                                                                 axes=(1, )))

        # Output the elementwise sum of the axiswise sum of matrices M1 ad M2 along the columns
        self.register_output('multiple_matrix_sum_along_0',
                             ot.sum(M1, M2, axes=(0, )))

        # Output the elementwise sum of the axiswise sum of matrices M1 ad M2 along the rows
        self.register_output('multiple_matrix_sum_along_1',
                             ot.sum(M1, M2, axes=(1, )))


prob = Problem()
prob.model = ExampleSum()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('single_vector_sum', prob['single_vector_sum'].shape)
print(prob['single_vector_sum'])
print('single_matrix_sum', prob['single_matrix_sum'].shape)
print(prob['single_matrix_sum'])
print('single_tensor_sum', prob['single_tensor_sum'].shape)
print(prob['single_tensor_sum'])
print('multiple_vector_sum', prob['multiple_vector_sum'].shape)
print(prob['multiple_vector_sum'])
print('multiple_matrix_sum', prob['multiple_matrix_sum'].shape)
print(prob['multiple_matrix_sum'])
print('multiple_tensor_sum', prob['multiple_tensor_sum'].shape)
print(prob['multiple_tensor_sum'])
print('single_matrix_sum_along_0', prob['single_matrix_sum_along_0'].shape)
print(prob['single_matrix_sum_along_0'])
print('single_matrix_sum_along_1', prob['single_matrix_sum_along_1'].shape)
print(prob['single_matrix_sum_along_1'])
print('multiple_matrix_sum_along_0', prob['multiple_matrix_sum_along_0'].shape)
print(prob['multiple_matrix_sum_along_0'])
print('multiple_matrix_sum_along_1', prob['multiple_matrix_sum_along_1'].shape)
print(prob['multiple_matrix_sum_along_1'])
