from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExamplePNorm(Group):
    """
    :param var: axis_free_pnorm
    :param var: axiswise_pnorm
    """
    def setup(self):

        # Shape of the tensor
        shape = (2, 3, 4, 5)

        # Number of elements in the tensor
        num_of_elements = np.prod(shape)

        # Creating a numpy tensor with the desired shape and size
        tensor = np.arange(num_of_elements).reshape(shape)

        # Declaring in1 as input tensor
        in1 = self.declare_input('in1', val=tensor)

        # Computing the 6-norm on in1 without specifying an axis
        self.register_output('axis_free_pnorm', ot.pnorm(in1, pnorm_type=6))

        # Computing the 6-norm of in1 over the specified axes.
        self.register_output('axiswise_pnorm',
                             ot.pnorm(in1, axis=(1, 3), pnorm_type=6))


class ExampleTranspose(Group):
    """
    :param var: matrix_transpose
    :param var: tensor_transpose
    """
    def setup(self):

        # Declare mat as an input matrix with shape = (4, 2)
        mat = self.declare_input(
            'Mat',
            val=np.arange(4 * 2).reshape((4, 2)),
        )

        # Declare tens as an input tensor with shape = (4, 3, 2, 5)
        tens = self.declare_input(
            'Tens',
            val=np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2)),
        )

        # Compute the transpose of mat
        self.register_output('matrix_transpose', ot.transpose(mat))

        # Compute the transpose of tens
        self.register_output('tensor_transpose', ot.transpose(tens))


class ExampleDot(Group):
    """
    :param var: VecVecDot
    :param var: TenTenDotFirst
    :param var: TenTenDotLast
    """
    def setup(self):

        m = 3
        n = 4
        p = 5

        # Shape of the vectors
        vec_shape = (m, )

        # Shape of the tensors
        ten_shape = (m, n, p)

        # Values for the two vectors
        vec1 = np.arange(m)
        vec2 = np.arange(m, 2 * m)

        # Number of elements in the tensors
        num_ten_elements = np.prod(ten_shape)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements).reshape(ten_shape)
        ten2 = np.arange(num_ten_elements,
                         2 * num_ten_elements).reshape(ten_shape)

        # Adding the vectors and tensors to omtools
        vec1 = self.declare_input('vec1', val=vec1)
        vec2 = self.declare_input('vec2', val=vec2)

        ten1 = self.declare_input('ten1', val=ten1)
        ten2 = self.declare_input('ten2', val=ten2)

        # Vector-Vector Dot Product
        self.register_output('VecVecDot', ot.dot(vec1, vec2))

        # Tensor-Tensor Dot Product specifying the first axis
        self.register_output('TenTenDotFirst', ot.dot(ten1, ten2, axis=0))

        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('TenTenDotLast', ot.dot(ten1, ten2, axis=2))


class ExampleCross(Group):
    """
    :param var: VecVecCross
    :param var: TenTenCross
    """
    def setup(self):
        # Creating two vectors
        vecval1 = np.arange(3)
        vecval2 = np.arange(3) + 1

        vec1 = self.declare_input('vec1', val=vecval1)
        vec2 = self.declare_input('vec2', val=vecval2)

        # Vector-Vector Cross Product
        self.register_output('VecVecCross', ot.cross(vec1, vec2, axis=0))

        # Creating two tensors
        shape = (2, 5, 4, 3)
        num_elements = np.prod(shape)

        tenval1 = np.arange(num_elements).reshape(shape)
        tenval2 = np.arange(num_elements).reshape(shape) + 6

        ten1 = self.declare_input('ten1', val=tenval1)
        ten2 = self.declare_input('ten2', val=tenval2)

        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('TenTenCross', ot.cross(ten1, ten2, axis=3))


class ExampleInnerProduct(Group):
    """
    :param var: VecVecInner
    :param var: TenVecInner
    :param var: TenTenInner
    """
    def setup(self):

        m = 3
        n = 4
        p = 5

        # Shape of the vectors
        vec_shape = (m, )

        # Shape of the tensors
        ten_shape = (m, n, p)

        # Values for the two vectors
        vec1 = np.arange(m)
        vec2 = np.arange(m, 2 * m)

        # Number of elements in the tensors
        num_ten_elements = np.prod(ten_shape)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements).reshape(ten_shape)
        ten2 = np.arange(num_ten_elements,
                         2 * num_ten_elements).reshape(ten_shape)

        # Adding the vectors and tensors to omtools
        vec1 = self.declare_input('vec1', val=vec1)
        vec2 = self.declare_input('vec2', val=vec2)

        ten1 = self.declare_input('ten1', val=ten1)
        ten2 = self.declare_input('ten2', val=ten2)

        # Vector-Vector Inner Product
        self.register_output('VecVecInner', ot.inner(vec1, vec2))

        # Tensor-Vector Inner Product specifying the first axis for
        # Vector and Tensor
        self.register_output(
            'TenVecInner',
            ot.inner(ten1, vec1, axes=([0], [0])),
        )

        # Tensor-Tensor Inner Product specifying the first and last axes
        self.register_output(
            'TenTenInner',
            ot.inner(ten1, ten2, axes=([0, 2], [0, 2])),
        )


class ExampleOuterProduct(Group):
    """
    :param var: VecVecOuter
    :param var: TenVecOuter
    :param var: TenTenOuter
    """
    def setup(self):

        m = 3
        n = 4
        p = 5

        # Shape of the vectors
        vec_shape = (m, )

        # Shape of the tensors
        ten_shape = (m, n, p)

        # Values for the two vectors
        vec1 = np.arange(m)
        vec2 = np.arange(m, 2 * m)

        # Number of elements in the tensors
        num_ten_elements = np.prod(ten_shape)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements).reshape(ten_shape)
        ten2 = np.arange(num_ten_elements,
                         2 * num_ten_elements).reshape(ten_shape)

        # Adding the vectors and tensors to omtools
        vec1 = self.declare_input('vec1', val=vec1)
        vec2 = self.declare_input('vec2', val=vec2)

        ten1 = self.declare_input('ten1', val=ten1)
        ten2 = self.declare_input('ten2', val=ten2)

        # Vector-Vector Outer Product
        self.register_output('VecVecOuter', ot.outer(vec1, vec2))

        # Tensor-Vector Outer Product specifying the first axis for Vector and Tensor
        self.register_output('TenVecOuter', ot.outer(ten1, vec1))

        # Tensor-Tensor Outer Product specifying the first and last axes
        self.register_output('TenTenOuter', ot.outer(ten1, ten2))


class ExampleMatrixVectorProduct(Group):
    """
    :param var: MatVec
    """
    def setup(self):
        m = 3
        n = 4

        # Shape of the first matrix (3,2)
        shape1 = (m, n)

        # Shape of the second matrix (2,4)
        shape2 = (n, )

        # Creating the values of both matrices
        val1 = np.arange(m * n).reshape(shape1)
        val2 = np.arange(n).reshape(shape2)

        # Declaring the input matrix and input vector
        mat1 = self.declare_input('mat1', val=val1)
        vec1 = self.declare_input('vec1', val=val2)

        # Creating the output for matrix-vector multiplication
        self.register_output('MatVec', ot.matvec(mat1, vec1))


class ExampleMatrixMatrixProduct(Group):
    """
    :param var: MatMat
    :param var: MatVec
    """
    def setup(self):
        m = 3
        n = 2
        p = 4

        # Shape of the first matrix (3,2)
        shape1 = (m, n)

        # Shape of the second matrix (2,4)
        shape2 = (n, p)

        # Creating the values of both matrices
        val1 = np.arange(m * n).reshape(shape1)
        val2 = np.arange(n * p).reshape(shape2)

        # Creating the values for the vector
        val3 = np.arange(n)

        # Declaring the two input matrices as mat1 and mat2
        mat1 = self.declare_input('mat1', val=val1)
        mat2 = self.declare_input('mat2', val=val2)

        # Declaring the input vector of size (n,)
        vec1 = self.declare_input('vec1', val=val3)

        # Creating the output for matrix multiplication
        self.register_output('MatMat', ot.matmat(mat1, mat2))

        # Creating the output for a matrix multiplied by a vector
        self.register_output('MatVec', ot.matmat(mat1, vec1))


class ExampleRotationMatrix(Group):
    """
    :param var: scalar_Rot_x
    :param var: scalar_Rot_y
    :param var: tensor1_Rot_x
    :param var: tensor2_Rot_x
    """
    def setup(self):

        # Shape of a random tensor rotation matrix
        shape = (2, 3, 4)

        num_elements = np.prod(shape)

        # Tensor of angles in radians
        angle_val1 = np.repeat(np.pi / 3, num_elements).reshape(shape)

        angle_val2 = np.repeat(
            np.pi / 3, num_elements) + 2 * np.pi * np.arange(num_elements)

        angle_val2 = angle_val2.reshape(shape)

        angle_val3 = np.pi / 3

        # Adding the tensor as an input
        angle_tensor1 = self.declare_input('tensor1', val=angle_val1)

        angle_tensor2 = self.declare_input('tensor2', val=angle_val2)

        angle_scalar = self.declare_input('scalar', val=angle_val3)

        # Rotation in the x-axis for scalar
        self.register_output('scalar_Rot_x', ot.rotmat(angle_scalar, axis='x'))

        # Rotation in the y-axis for scalar
        self.register_output('scalar_Rot_y', ot.rotmat(angle_scalar, axis='y'))

        # Rotation in the x-axis for tensor1
        self.register_output('tensor1_Rot_x', ot.rotmat(angle_tensor1,
                                                        axis='x'))

        # Rotation in the x-axis for tensor2
        self.register_output('tensor2_Rot_x', ot.rotmat(angle_tensor2,
                                                        axis='x'))


class ErrorPnormTypeNotPositive(Group):
    def setup(self):

        # Shape of the tensor
        shape = (2, 3, 4, 5)

        # Number of elements in the tensor
        num_of_elements = np.prod(shape)

        # Creating a numpy tensor with the desired shape and size
        tensor = np.arange(num_of_elements).reshape(shape)

        # Declaring in1 as input tensor
        in1 = self.declare_input('in1', val=tensor)

        # Computing the 6-norm on in1 without specifying an axis
        self.register_output('axis_free_pnorm', ot.pnorm(in1, pnorm_type=-2))


class ErrorPnormTypeNotEven(Group):
    def setup(self):

        # Shape of the tensor
        shape = (2, 3, 4, 5)

        # Number of elements in the tensor
        num_of_elements = np.prod(shape)

        # Creating a numpy tensor with the desired shape and size
        tensor = np.arange(num_of_elements).reshape(shape)

        # Declaring in1 as input tensor
        in1 = self.declare_input('in1', val=tensor)

        # Computing the 6-norm on in1 without specifying an axis
        self.register_output('axis_free_pnorm', ot.pnorm(in1, pnorm_type=3))


class ErrorDotVecDifferentShapes(Group):
    def setup(self):
        m = 3
        n = 4
        p = 5

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


class ErrorDotTenDifferentShapes(Group):
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