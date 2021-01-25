from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleVectorVector(Group):
    """
    :param var: vec1
    :param var: vec2
    :param var: VecVecCross
    """
    def setup(self):
        # Creating two vectors
        vecval1 = np.arange(3)
        vecval2 = np.arange(3) + 1

        vec1 = self.declare_input('vec1', val=vecval1)
        vec2 = self.declare_input('vec2', val=vecval2)

        # Vector-Vector Cross Product
        self.register_output('VecVecCross', ot.cross(vec1, vec2, axis=0))


class ExampleTensorTensor(Group):
    """
    :param var: ten1
    :param var: ten2
    :param var: TenTenCross
    """
    def setup(self):
        # Creating two tensors
        shape = (2, 5, 4, 3)
        num_elements = np.prod(shape)

        tenval1 = np.arange(num_elements).reshape(shape)
        tenval2 = np.arange(num_elements).reshape(shape) + 6

        ten1 = self.declare_input('ten1', val=tenval1)
        ten2 = self.declare_input('ten2', val=tenval2)

        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('TenTenCross', ot.cross(ten1, ten2, axis=3))


class ErrorDifferentShapes(Group):
    def setup(self):
        # Creating two tensors
        shape1 = (2, 5, 4, 3)
        shape2 = (7, 5, 6, 3)
        num_elements1 = np.prod(shape1)
        num_elements2 = np.prod(shape2)

        tenval1 = np.arange(num_elements1).reshape(shape1)
        tenval2 = np.arange(num_elements2).reshape(shape2) + 6

        ten1 = self.declare_input('ten1', val=tenval1)
        ten2 = self.declare_input('ten2', val=tenval2)

        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('TenTenCross', ot.cross(ten1, ten2, axis=3))


class ErrorIncorrectAxisIndex(Group):
    def setup(self):
        # Creating two tensors
        shape = (2, 5, 4, 3)
        num_elements = np.prod(shape)

        tenval1 = np.arange(num_elements).reshape(shape)
        tenval2 = np.arange(num_elements).reshape(shape) + 6

        ten1 = self.declare_input('ten1', val=tenval1)
        ten2 = self.declare_input('ten2', val=tenval2)

        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('TenTenCross', ot.cross(ten1, ten2, axis=2))