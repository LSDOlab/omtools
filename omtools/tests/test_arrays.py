from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_min():
    import omtools.examples.valid.ex_arrays_min as example

    m = 2
    n = 3
    o = 4
    p = 5
    q = 6

    tensor_shape = (m, n, o, p, q)
    num_of_elements = np.prod(tensor_shape)
    tensor = np.arange(num_of_elements).reshape(tensor_shape)

    # AXISWISE MIN
    desired_output = np.amin(tensor, axis=1)
    np.testing.assert_almost_equal(example.prob['AxiswiseMin'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_AxiswiseMin'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

    # SCALAR MIN
    desired_output = np.min(tensor)
    np.testing.assert_almost_equal(example.prob['ScalarMin'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_ScalarMin'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

    # ELEMENTWISE MIN
    m = 2
    n = 3
    # Shape of the three tensors is (2,3)
    shape = (m, n)

    tensor1 = np.array([[1, 5, -8], [10, -3, -5]])
    tensor2 = np.array([[2, 6, 9], [-1, 2, 4]])

    desired_output = np.minimum(tensor1, tensor2)
    np.testing.assert_almost_equal(example.prob['ElementwiseMin'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_ElementwiseMin'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_max():
    import omtools.examples.valid.ex_arrays_max as example

    m = 2
    n = 3
    o = 4
    p = 5
    q = 6

    tensor_shape = (m, n, o, p, q)
    num_of_elements = np.prod(tensor_shape)
    tensor = np.arange(num_of_elements).reshape(tensor_shape)

    # AXISWISE MAX
    desired_output = np.amax(tensor, axis=1)
    np.testing.assert_almost_equal(example.prob['AxiswiseMax'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_AxiswiseMax'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

    # SCALAR MAX
    desired_output = np.max(tensor)
    np.testing.assert_almost_equal(example.prob['ScalarMax'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_ScalarMax'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

    # ELEMENTWISE MAX
    m = 2
    n = 3
    # Shape of the three tensors is (2,3)
    shape = (m, n)

    tensor1 = np.array([[1, 5, -8], [10, -3, -5]])
    tensor2 = np.array([[2, 6, 9], [-1, 2, 4]])

    desired_output = np.maximum(tensor1, tensor2)
    np.testing.assert_almost_equal(example.prob['ElementwiseMax'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_ElementwiseMax'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_reshape():
    import omtools.examples.valid.ex_arrays_reshape as example

    i = 2
    j = 3
    k = 4
    l = 5
    shape = (i, j, k, l)

    tensor = np.arange(np.prod(shape)).reshape(shape)
    vector = np.arange(np.prod(shape))

    # TENSOR TO VECTOR
    desired_output = vector
    np.testing.assert_almost_equal(example.prob['reshape_tensor2vector'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_reshape_tensor2vector'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

    # VECTOR TO TENSOR
    desired_output = tensor

    np.testing.assert_almost_equal(example.prob['reshape_vector2tensor'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_reshape_vector2tensor'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)


def test_matrix_reorder():
    import omtools.examples.valid.ex_arrays_reorder_axes as example
    val = np.arange(4 * 2).reshape((4, 2))
    desired_output = np.transpose(val)

    np.testing.assert_almost_equal(example.prob['axes_reordered_matrix'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_axes_reordered_matrix'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

    val = np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2))
    desired_output = np.transpose(val, [3, 1, 2, 0])

    np.testing.assert_almost_equal(example.prob['axes_reordered_tensor'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_axes_reordered_tensor'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_sum():
    import omtools.examples.valid.ex_arrays_sum as example

    n = 3
    m = 6
    p = 7
    q = 10

    v1 = np.arange(n)
    v2 = np.arange(n, 2 * n)

    M1 = np.arange(n * m).reshape((n, m))
    M2 = np.arange(n * m, 2 * n * m).reshape((n, m))

    T1 = np.arange(n * m * p * q).reshape((n, m, p, q))
    T2 = np.arange(n * m * p * q, 2 * n * m * p * q).reshape((n, m, p, q))

    # TEST FULL ARRAY SUM (ERROR WITH TENSOR SUM PARTIALS)
    desired_vector_sum = np.sum(v1)
    desired_matrix_sum = np.sum(M1)
    desired_tensor_sum = np.sum(T1)

    np.testing.assert_almost_equal(example.prob['single_vector_sum'],
                                   desired_vector_sum)

    np.testing.assert_almost_equal(example.prob['single_tensor_sum'],
                                   desired_tensor_sum)

    partials_error_vector_sum = example.prob.check_partials(
        includes=['comp_single_vector_sum'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_vector_sum, atol=1.e-6, rtol=1.e-6)

    partials_error_tensor_sum = example.prob.check_partials(
        includes=['comp_single_tensor_sum'],
        out_stream=None,
        compact_print=True)
    # assert_check_partials(partials_error_tensor_sum, atol=1.e-6, rtol=1.e-6)

    # TEST MUTLIPLE ARRAY SUM

    desired_vector_sum = v1 + v2
    desired_matrix_sum = M1 + M2
    desired_tensor_sum = T1 + T2

    np.testing.assert_almost_equal(example.prob['multiple_vector_sum'],
                                   desired_vector_sum)

    np.testing.assert_almost_equal(example.prob['multiple_tensor_sum'],
                                   desired_tensor_sum)

    partials_error_vector_sum = example.prob.check_partials(
        includes=['comp_multiple_vector_sum'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_vector_sum, atol=1.e-6, rtol=1.e-6)

    partials_error_tensor_sum = example.prob.check_partials(
        includes=['comp_multiple_tensor_sum'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_tensor_sum, atol=1.e-5, rtol=1.e-5)

    # TEST MATRIX SUM AXISWISE

    desired_single_matrix_sum_axis_0 = np.sum(M1, axis=0)
    desired_single_matrix_sum_axis_1 = np.sum(M1, axis=1)

    desired_multiple_matrix_sum_axis_0 = np.sum(M1 + M2, axis=0)
    desired_multiple_matrix_sum_axis_1 = np.sum(M1 + M2, axis=1)

    np.testing.assert_almost_equal(example.prob['single_matrix_sum_along_0'],
                                   desired_single_matrix_sum_axis_0)
    np.testing.assert_almost_equal(example.prob['single_matrix_sum_along_1'],
                                   desired_single_matrix_sum_axis_1)

    np.testing.assert_almost_equal(example.prob['multiple_matrix_sum_along_0'],
                                   desired_multiple_matrix_sum_axis_0)
    np.testing.assert_almost_equal(example.prob['multiple_matrix_sum_along_1'],
                                   desired_multiple_matrix_sum_axis_1)

    partials_error_single_matrix_axis_0 = example.prob.check_partials(
        includes=['comp_single_matrix_sum_along_0'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_single_matrix_axis_0,
                          atol=1.e-6,
                          rtol=1.e-6)

    partials_error_single_matrix_axis_1 = example.prob.check_partials(
        includes=['comp_single_matrix_sum_along_1'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_single_matrix_axis_1,
                          atol=1.e-6,
                          rtol=1.e-6)

    partials_error_multiple_matrix_axis_0 = example.prob.check_partials(
        includes=['comp_multiple_matrix_sum_along_0'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_multiple_matrix_axis_0,
                          atol=1.e-6,
                          rtol=1.e-6)

    partials_error_multiple_matrix_axis_1 = example.prob.check_partials(
        includes=['comp_multiple_matrix_sum_along_1'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_multiple_matrix_axis_1,
                          atol=1.e-6,
                          rtol=1.e-6)


def test_average():
    import omtools.examples.valid.ex_arrays_average as example

    n = 3
    m = 6
    p = 7
    q = 10

    v1 = np.arange(n)
    v2 = np.arange(n, 2 * n)

    M1 = np.arange(n * m).reshape((n, m))
    M2 = np.arange(n * m, 2 * n * m).reshape((n, m))

    T1 = np.arange(n * m * p * q).reshape((n, m, p, q))
    T2 = np.arange(n * m * p * q, 2 * n * m * p * q).reshape((n, m, p, q))

    # TEST FULL ARRAY AVERAGE
    desired_vector_average = np.average(v1)
    desired_matrix_average = np.average(M1)
    desired_tensor_average = np.average(T1)

    np.testing.assert_almost_equal(example.prob['single_vector_average'],
                                   desired_vector_average)
    np.testing.assert_almost_equal(example.prob['single_matrix_average'],
                                   desired_matrix_average)
    np.testing.assert_almost_equal(example.prob['single_tensor_average'],
                                   desired_tensor_average)

    partials_error_vector_average = example.prob.check_partials(
        includes=['comp_single_vector_average'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_vector_average,
                          atol=1.e-6,
                          rtol=1.e-6)

    partials_error_matrix_average = example.prob.check_partials(
        includes=['comp_single_matrix_average'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_matrix_average,
                          atol=1.e-6,
                          rtol=1.e-6)

    partials_error_tensor_average = example.prob.check_partials(
        includes=['comp_single_tensor_average'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_tensor_average,
                          atol=1.e-5,
                          rtol=1.e-5)

    # TEST MULTIPLE ARRAY AVERAGE
    desired_vector_average = (v1 + v2) / 2.
    desired_matrix_average = (M1 + M2) / 2.
    desired_tensor_average = (T1 + T2) / 2.

    np.testing.assert_almost_equal(example.prob['multiple_vector_average'],
                                   desired_vector_average)
    np.testing.assert_almost_equal(example.prob['multiple_matrix_average'],
                                   desired_matrix_average)
    np.testing.assert_almost_equal(example.prob['multiple_tensor_average'],
                                   desired_tensor_average)

    partials_error_vector_average = example.prob.check_partials(
        includes=['comp_multiple_vector_average'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_vector_average,
                          atol=1.e-6,
                          rtol=1.e-6)

    partials_error_matrix_average = example.prob.check_partials(
        includes=['comp_multiple_matrix_average'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_matrix_average,
                          atol=1.e-6,
                          rtol=1.e-6)

    partials_error_tensor_average = example.prob.check_partials(
        includes=['comp_multiple_tensor_average'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_tensor_average,
                          atol=1.e-5,
                          rtol=1.e-5)

    # TEST MATRIX AVERAGE AXISWISE

    desired_single_matrix_average_axis_0 = np.average(M1, axis=0)
    desired_single_matrix_average_axis_1 = np.average(M1, axis=1)

    desired_multiple_matrix_average_axis_0 = np.average((M1 + M2) / 2., axis=0)
    desired_multiple_matrix_average_axis_1 = np.average((M1 + M2) / 2., axis=1)

    np.testing.assert_almost_equal(
        example.prob['single_matrix_average_along_0'],
        desired_single_matrix_average_axis_0)
    np.testing.assert_almost_equal(
        example.prob['single_matrix_average_along_1'],
        desired_single_matrix_average_axis_1)

    np.testing.assert_almost_equal(
        example.prob['multiple_matrix_average_along_0'],
        desired_multiple_matrix_average_axis_0)
    np.testing.assert_almost_equal(
        example.prob['multiple_matrix_average_along_1'],
        desired_multiple_matrix_average_axis_1)

    partials_error_single_matrix_axis_0 = example.prob.check_partials(
        includes=['comp_single_matrix_average_along_0'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_single_matrix_axis_0,
                          atol=1.e-6,
                          rtol=1.e-6)

    partials_error_single_matrix_axis_1 = example.prob.check_partials(
        includes=['comp_single_matrix_average_along_1'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_single_matrix_axis_1,
                          atol=1.e-6,
                          rtol=1.e-6)

    partials_error_multiple_matrix_axis_0 = example.prob.check_partials(
        includes=['comp_multiple_matrix_average_along_0'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_multiple_matrix_axis_0,
                          atol=1.e-6,
                          rtol=1.e-6)

    partials_error_multiple_matrix_axis_1 = example.prob.check_partials(
        includes=['comp_multiple_matrix_average_along_1'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error_multiple_matrix_axis_1,
                          atol=1.e-6,
                          rtol=1.e-6)
