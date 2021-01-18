from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_pnorm():
    import omtools.examples.valid.ex_linear_algebra_p_norm as example

    i = 2
    j = 3
    k = 4
    l = 5
    shape = (i, j, k, l)
    pnorm_type = 6

    val = np.arange(np.prod(shape)).reshape(shape)

    desired_output = np.linalg.norm(val.flatten(), ord=pnorm_type)
    np.testing.assert_almost_equal(example.prob['axis_free_pnorm'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_axis_free_pnorm'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)

    axis = (1, 3)
    desired_output = np.sum(val**pnorm_type, axis=axis)**(1 / pnorm_type)

    np.testing.assert_almost_equal(example.prob['axiswise_pnorm'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_axiswise_pnorm'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)


def test_transpose():
    import omtools.examples.valid.ex_linear_algebra_transpose as example

    # MATRIX TRANSPOSE
    val = np.arange(4 * 2).reshape((4, 2))
    desired_output = np.transpose(val)

    np.testing.assert_almost_equal(example.prob['matrix_transpose'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_matrix_transpose'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

    # TENSOR TRANSPOSE
    val = np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2))
    desired_output = np.transpose(val)

    np.testing.assert_almost_equal(example.prob['tensor_transpose'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_tensor_transpose'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_dot():
    import omtools.examples.valid.ex_linear_algebra_dot as example

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
    ten2 = np.arange(num_ten_elements, 2 * num_ten_elements).reshape(ten_shape)

    # VECTOR VECTOR
    desired_output = np.dot(vec1, vec2)
    np.testing.assert_almost_equal(example.prob['VecVecDot'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_VecVecDot'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

    # TENSOR TENSOR
    desired_output = np.tensordot(ten1, ten2, axes=([0], [0]))
    np.testing.assert_almost_equal(example.prob['TenTenDotFirst'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_TenTenDotFirst'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)

    desired_output = np.tensordot(ten1, ten2, axes=([2], [2]))
    np.testing.assert_almost_equal(example.prob['TenTenDotLast'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_TenTenDotLast'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)


def test_cross():
    import omtools.examples.valid.ex_linear_algebra_cross as example

    vec1 = np.arange(3)
    vec2 = np.arange(3) + 1

    shape = (2, 5, 4, 3)
    num_elements = np.prod(shape)

    ten1 = np.arange(num_elements).reshape(shape)
    ten2 = np.arange(num_elements).reshape(shape) + 6

    desired_output = np.cross(vec1, vec2)
    np.testing.assert_almost_equal(example.prob['VecVecCross'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_VecVecCross'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

    desired_output = np.cross(ten1, ten2, axis=3)
    np.testing.assert_almost_equal(example.prob['TenTenCross'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_TenTenCross'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)


def test_inner():
    import omtools.examples.valid.ex_linear_algebra_inner_product as example

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
    ten2 = np.arange(num_ten_elements, 2 * num_ten_elements).reshape(ten_shape)

    # VECTOR VECTOR INNER
    desired_output = np.dot(vec1, vec2)
    np.testing.assert_almost_equal(example.prob['VecVecInner'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_VecVecInner'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

    # TENSOR VECTOR INNER
    desired_output = np.tensordot(ten1, vec1, axes=([0], [0]))
    np.testing.assert_almost_equal(example.prob['TenVecInner'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_TenVecInner'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

    # TENSOR TENSOR INNER
    desired_output = np.tensordot(ten1, ten2, axes=([0, 2], [0, 2]))
    np.testing.assert_almost_equal(example.prob['TenTenInner'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_TenTenInner'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)


def test_outer():
    import omtools.examples.valid.ex_linear_algebra_outer_product as example

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
    ten2 = np.arange(num_ten_elements, 2 * num_ten_elements).reshape(ten_shape)

    # VEC VEC OUTER
    desired_output = np.outer(vec1, vec2)
    np.testing.assert_almost_equal(example.prob['VecVecOuter'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_VecVecOuter'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

    # TENSOR VECTOR OUTER
    desired_output = np.einsum('ijk,l->ijkl', ten1, vec1)
    np.testing.assert_almost_equal(example.prob['TenVecOuter'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_TenVecOuter'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

    # TENSOR TENSOR OUTER
    desired_output = np.einsum('ijk,lmn->ijklmn', ten1, ten2)
    np.testing.assert_almost_equal(example.prob['TenTenOuter'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_TenTenOuter'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)


def test_matrix_matrix_multiplication():
    import omtools.examples.valid.ex_linear_algebra_matrix_matrix_product as example

    m = 3
    n = 2
    p = 4

    # Shape of the first matrix (3,2)
    shape1 = (m, n)

    # Shape of the second matrix (2,4)
    shape2 = (n, p)

    # Creating the values of both matrices
    mat1 = np.arange(m * n).reshape(shape1)
    mat2 = np.arange(n * p).reshape(shape2)

    # Creating the values for the vector
    vec1 = np.arange(n)

    # MATRIX MATRIX MULTIPLICATION
    desired_output = np.matmul(mat1, mat2)
    np.testing.assert_almost_equal(example.prob['MatMat'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_MatMat'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_matrix_vector_multiplication():
    import omtools.examples.valid.ex_linear_algebra_matrix_vector_product as example

    m = 3
    n = 4

    # Shape of the first matrix (3,2)
    shape1 = (m, n)

    # Shape of the vector (4,)
    shape2 = (n, )

    # Creating the matrix
    mat1 = np.arange(m * n).reshape(shape1)

    # Creating the vector
    vec1 = np.arange(n).reshape(shape2)

    # MATRIX VECTOR MULTIPLICATION
    desired_output = np.matmul(mat1, vec1)
    np.testing.assert_almost_equal(example.prob['MatVec'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_MatVec'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)
