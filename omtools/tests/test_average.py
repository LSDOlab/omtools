from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import omtools.examples.ex_average_comp as example

# TODO: figure out the mismatch in openMDAO partial errors between check_partials and assert_partials

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

def test_full_array_average():
    desired_vector_average = np.average(v1)
    desired_matrix_average = np.average(M1)
    desired_tensor_average = np.average(T1)

    np.testing.assert_almost_equal(example.prob['single_vector_average'], desired_vector_average)
    np.testing.assert_almost_equal(example.prob['single_matrix_average'], desired_matrix_average)
    np.testing.assert_almost_equal(example.prob['single_tensor_average'], desired_tensor_average)

    partials_error_vector_average = example.prob.check_partials(includes=['comp_single_vector_average'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error_vector_average, atol=1.e-6, rtol=1.e-6)
    
    partials_error_matrix_average = example.prob.check_partials(includes=['comp_single_matrix_average'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error_matrix_average, atol=1.e-6, rtol=1.e-6)

    partials_error_tensor_average = example.prob.check_partials(includes=['comp_single_tensor_average'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error_tensor_average, atol=1.e-5, rtol=1.e-5)

def test_multiple_array_average():
    desired_vector_average = (v1 + v2) / 2.
    desired_matrix_average = (M1 + M2) / 2.
    desired_tensor_average = (T1 + T2) / 2.

    np.testing.assert_almost_equal(example.prob['multiple_vector_average'], desired_vector_average)
    np.testing.assert_almost_equal(example.prob['multiple_matrix_average'], desired_matrix_average)
    np.testing.assert_almost_equal(example.prob['multiple_tensor_average'], desired_tensor_average)

    partials_error_vector_average = example.prob.check_partials(includes=['comp_multiple_vector_average'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error_vector_average, atol=1.e-6, rtol=1.e-6)
    
    partials_error_matrix_average = example.prob.check_partials(includes=['comp_multiple_matrix_average'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error_matrix_average, atol=1.e-6, rtol=1.e-6)

    partials_error_tensor_average = example.prob.check_partials(includes=['comp_multiple_tensor_average'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error_tensor_average, atol=1.e-5, rtol=1.e-5)

def test_matrix_average_axiswise():
    desired_single_matrix_average_axis_0 = np.average(M1, axis = 0)
    desired_single_matrix_average_axis_1 = np.average(M1, axis = 1)

    desired_multiple_matrix_average_axis_0 = np.average((M1 + M2) / 2. , axis = 0)
    desired_multiple_matrix_average_axis_1 = np.average((M1 + M2) / 2. , axis = 1)

    np.testing.assert_almost_equal(example.prob['single_matrix_average_along_0'], desired_single_matrix_average_axis_0)
    np.testing.assert_almost_equal(example.prob['single_matrix_average_along_1'], desired_single_matrix_average_axis_1)

    np.testing.assert_almost_equal(example.prob['multiple_matrix_average_along_0'], desired_multiple_matrix_average_axis_0)
    np.testing.assert_almost_equal(example.prob['multiple_matrix_average_along_1'], desired_multiple_matrix_average_axis_1)

    partials_error_single_matrix_axis_0 = example.prob.check_partials(includes=['comp_single_matrix_average_along_0'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error_single_matrix_axis_0, atol=1.e-6, rtol=1.e-6)
    
    partials_error_single_matrix_axis_1= example.prob.check_partials(includes=['comp_single_matrix_average_along_1'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error_single_matrix_axis_1, atol=1.e-6, rtol=1.e-6)

    partials_error_multiple_matrix_axis_0 = example.prob.check_partials(includes=['comp_multiple_matrix_average_along_0'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error_multiple_matrix_axis_0, atol=1.e-6, rtol=1.e-6)

    partials_error_multiple_matrix_axis_1 = example.prob.check_partials(includes=['comp_multiple_matrix_average_along_1'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error_multiple_matrix_axis_1, atol=1.e-6, rtol=1.e-6)