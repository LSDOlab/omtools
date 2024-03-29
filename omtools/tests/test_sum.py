from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_sum_single_vector():
    import omtools.examples.valid.ex_sum_single_vector as example

    n = 3

    v1 = np.arange(n)

    desired_vector_sum = np.sum(v1)

    np.testing.assert_almost_equal(example.prob['single_vector_sum'],
                                   desired_vector_sum)

    partials_error_vector_sum = example.prob.check_partials(
        includes=['comp_single_vector_sum'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_vector_sum, atol=1.e-6, rtol=1.e-6)


def test_sum_single_matrix():
    import omtools.examples.valid.ex_sum_single_matrix as example

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))

    desired_matrix_sum = np.sum(M1)

    np.testing.assert_almost_equal(example.prob['single_matrix_sum'],
                                   desired_matrix_sum)

    partials_error_vector_sum = example.prob.check_partials(
        includes=['comp_single_matrix_sum'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_vector_sum, atol=1.e-6, rtol=1.e-6)


def test_sum_single_tensor():
    import omtools.examples.valid.ex_sum_single_tensor as example

    n = 3
    m = 4
    p = 5
    q = 6

    T1 = np.arange(n * m * p * q).reshape((n, m, p, q))

    desired_tensor_sum = np.sum(T1)

    np.testing.assert_almost_equal(example.prob['single_tensor_sum'],
                                   desired_tensor_sum)

    partials_error_tensor_sum = example.prob.check_partials(
        includes=['comp_single_tensor_sum'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_tensor_sum, atol=1.e-5, rtol=1.e-5)


def test_sum_multiple_vector():
    import omtools.examples.valid.ex_sum_multiple_vector as example

    n = 3

    v1 = np.arange(n)
    v2 = np.arange(n, 2 * n)

    desired_vector_sum = v1 + v2

    np.testing.assert_almost_equal(example.prob['multiple_vector_sum'],
                                   desired_vector_sum)

    partials_error_vector_sum = example.prob.check_partials(
        includes=['comp_multiple_vector_sum'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_vector_sum, atol=1.e-6, rtol=1.e-6)


def test_sum_multiple_matrix():
    import omtools.examples.valid.ex_sum_multiple_matrix as example

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))
    M2 = np.arange(n * m, 2 * n * m).reshape((n, m))

    desired_matrix_sum = M1 + M2

    np.testing.assert_almost_equal(example.prob['multiple_matrix_sum'],
                                   desired_matrix_sum)

    partials_error_matrix_sum = example.prob.check_partials(
        includes=['comp_multiple_matrix_sum'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_matrix_sum, atol=1.e-6, rtol=1.e-6)


def test_sum_multiple_tensor():
    import omtools.examples.valid.ex_sum_multiple_tensor as example

    n = 3
    m = 6
    p = 7
    q = 10

    T1 = np.arange(n * m * p * q).reshape((n, m, p, q))
    T2 = np.arange(n * m * p * q, 2 * n * m * p * q).reshape((n, m, p, q))

    desired_tensor_sum = T1 + T2

    np.testing.assert_almost_equal(example.prob['multiple_tensor_sum'],
                                   desired_tensor_sum)

    partials_error_tensor_sum = example.prob.check_partials(
        includes=['comp_multiple_tensor_sum'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_tensor_sum, atol=1.e-5, rtol=1.e-5)


def test_sum_single_matrix_along0():
    import omtools.examples.valid.ex_sum_single_matrix_along0 as example

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))

    desired_single_matrix_sum_axis_0 = np.sum(M1, axis=0)

    np.testing.assert_almost_equal(example.prob['single_matrix_sum_along_0'],
                                   desired_single_matrix_sum_axis_0)

    partials_error_single_matrix_axis_0 = example.prob.check_partials(
        includes=['comp_single_matrix_sum_along_0'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_single_matrix_axis_0,
                          atol=1.e-6,
                          rtol=1.e-6)


def test_sum_single_matrix_along1():
    import omtools.examples.valid.ex_sum_single_matrix_along1 as example

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))

    desired_single_matrix_sum_axis_1 = np.sum(M1, axis=1)

    np.testing.assert_almost_equal(example.prob['single_matrix_sum_along_1'],
                                   desired_single_matrix_sum_axis_1)

    partials_error_single_matrix_axis_1 = example.prob.check_partials(
        includes=['comp_single_matrix_sum_along_1'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_single_matrix_axis_1,
                          atol=1.e-6,
                          rtol=1.e-6)


def test_sum_multiple_matrix_along0():
    import omtools.examples.valid.ex_sum_multiple_matrix_along0 as example

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))
    M2 = np.arange(n * m, 2 * n * m).reshape((n, m))

    desired_multiple_matrix_sum_axis_0 = np.sum(M1 + M2, axis=0)

    np.testing.assert_almost_equal(example.prob['multiple_matrix_sum_along_0'],
                                   desired_multiple_matrix_sum_axis_0)

    partials_error_multiple_matrix_axis_0 = example.prob.check_partials(
        includes=['comp_multiple_matrix_sum_along_0'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_multiple_matrix_axis_0,
                          atol=1.e-6,
                          rtol=1.e-6)


def test_sum_multiple_matrix_along1():
    import omtools.examples.valid.ex_sum_multiple_matrix_along1 as example

    n = 3
    m = 6

    M1 = np.arange(n * m).reshape((n, m))
    M2 = np.arange(n * m, 2 * n * m).reshape((n, m))

    desired_multiple_matrix_sum_axis_1 = np.sum(M1 + M2, axis=1)

    np.testing.assert_almost_equal(example.prob['multiple_matrix_sum_along_1'],
                                   desired_multiple_matrix_sum_axis_1)

    partials_error_multiple_matrix_axis_1 = example.prob.check_partials(
        includes=['comp_multiple_matrix_sum_along_1'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_multiple_matrix_axis_1,
                          atol=1.e-6,
                          rtol=1.e-6)