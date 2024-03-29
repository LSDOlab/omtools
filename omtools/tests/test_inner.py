from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_vector_vector_inner():
    import omtools.examples.valid.ex_inner_vector_vector as example

    m = 3

    # Shape of the vectors
    vec_shape = (m, )

    # Values for the two vectors
    vec1 = np.arange(m)
    vec2 = np.arange(m, 2 * m)

    # VECTOR VECTOR INNER
    desired_output = np.dot(vec1, vec2)
    np.testing.assert_almost_equal(example.prob['VecVecInner'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_VecVecInner'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_tensor_vector_inner():
    import omtools.examples.valid.ex_inner_tensor_vector as example

    m = 3
    n = 4
    p = 5

    # Shape of the vectors
    vec_shape = (m, )

    # Values for the two vectors
    vec1 = np.arange(m)

    # Shape of the tensors
    ten_shape = (m, n, p)

    # Number of elements in the tensors
    num_ten_elements = np.prod(ten_shape)

    # Values for the two tensors
    ten1 = np.arange(num_ten_elements).reshape(ten_shape)
    ten2 = np.arange(num_ten_elements, 2 * num_ten_elements).reshape(ten_shape)

    # TENSOR VECTOR INNER
    desired_output = np.tensordot(ten1, vec1, axes=([0], [0]))
    np.testing.assert_almost_equal(example.prob['TenVecInner'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_TenVecInner'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_tensor_tensor_inner():
    import omtools.examples.valid.ex_inner_tensor_tensor as example

    m = 3
    n = 4
    p = 5

    # Shape of the tensors
    ten_shape = (m, n, p)

    # Number of elements in the tensors
    num_ten_elements = np.prod(ten_shape)

    # Values for the two tensors
    ten1 = np.arange(num_ten_elements).reshape(ten_shape)
    ten2 = np.arange(num_ten_elements, 2 * num_ten_elements).reshape(ten_shape)

    # TENSOR TENSOR INNER
    desired_output = np.tensordot(ten1, ten2, axes=([0, 2], [0, 2]))
    np.testing.assert_almost_equal(example.prob['TenTenInner'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_TenTenInner'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)
