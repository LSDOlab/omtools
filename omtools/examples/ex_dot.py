from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleVectorVector(Group):
    """
    :param var: vec1
    :param var: vec2
    :param var: VecVecDot
    """
    def setup(self):

        m = 3

        # Shape of the vectors
        vec_shape = (m, )

        # Values for the two vectors
        vec1 = np.arange(m)
        vec2 = np.arange(m, 2 * m)

        # Adding the vectors to omtools
        vec1 = self.declare_input('vec1', val=vec1)
        vec2 = self.declare_input('vec2', val=vec2)

        # Vector-Vector Dot Product
        self.register_output('VecVecDot', ot.dot(vec1, vec2))


class ExampleTensorTensorFirst(Group):
    """
    :param var: ten1    
    :param var: ten2
    :param var: TenTenDotFirst
    """
    def setup(self):

        m = 3
        n = 4
        p = 5

        # Shape of the tensors
        ten_shape = (m, n, p)

        # Number of elements in the tensors
        num_ten_elements = np.prod(ten_shape)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements).reshape(ten_shape)
        ten2 = np.arange(num_ten_elements,
                         2 * num_ten_elements).reshape(ten_shape)

        # Adding the tensors to omtools
        ten1 = self.declare_input('ten1', val=ten1)
        ten2 = self.declare_input('ten2', val=ten2)

        # Tensor-Tensor Dot Product specifying the first axis
        self.register_output('TenTenDotFirst', ot.dot(ten1, ten2, axis=0))


class ExampleTensorTensorLast(Group):
    """
    :param var: ten1
    :param var: ten2
    :param var: TenTenDotLast
    """
    def setup(self):

        m = 3
        n = 4
        p = 5

        # Shape of the tensors
        ten_shape = (m, n, p)

        # Number of elements in the tensors
        num_ten_elements = np.prod(ten_shape)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements).reshape(ten_shape)
        ten2 = np.arange(num_ten_elements,
                         2 * num_ten_elements).reshape(ten_shape)

        # Adding the tensors to omtools
        ten1 = self.declare_input('ten1', val=ten1)
        ten2 = self.declare_input('ten2', val=ten2)

        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('TenTenDotLast', ot.dot(ten1, ten2, axis=2))


class ErrorVecDifferentShapes(Group):
    def setup(self):
        m = 3
        n = 4

        # Shape of the vectors
        vec_shape = (m, )

        # Values for the two vectors
        vec1 = np.arange(m)
        vec2 = np.arange(n, 2 * n)

        # Adding the vectors and tensors to omtools
        vec1 = self.declare_input('vec1', val=vec1)
        vec2 = self.declare_input('vec2', val=vec2)

        # Vector-Vector Dot Product
        self.register_output('VecVecDot', ot.dot(vec1, vec2))


class ErrorTenDifferentShapes(Group):
    def setup(self):
        m = 3
        n = 4
        p = 5

        # Shape of the tensors
        ten_shape1 = (m, n, p)
        ten_shape2 = (n, n, m)

        # Number of elements in the tensors
        num_ten_elements1 = np.prod(ten_shape1)
        num_ten_elements2 = np.prod(ten_shape2)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements1).reshape(ten_shape1)
        ten2 = np.arange(num_ten_elements2,
                         2 * num_ten_elements2).reshape(ten_shape2)

        ten1 = self.declare_input('ten1', val=ten1)
        ten2 = self.declare_input('ten2', val=ten2)

        # Tensor-Tensor Dot Product specifying the first axis
        self.register_output('TenTenDotFirst', ot.dot(ten1, ten2, axis=0))