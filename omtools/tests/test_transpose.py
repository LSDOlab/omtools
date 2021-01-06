from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import omtools.examples.ex_transpose_comp as example

def test_matrix_transpose():
    val = np.arange(4 * 2).reshape((4, 2))
    desired_output = np.transpose(val)

    np.testing.assert_almost_equal(example.prob['matrix_transpose'], desired_output) 

    partials_error = example.prob.check_partials(includes=['comp_matrix_transpose'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

def test_tensor_transpose():
    val = np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2))
    desired_output = np.transpose(val)

    np.testing.assert_almost_equal(example.prob['tensor_transpose'], desired_output) 

    partials_error = example.prob.check_partials(includes=['comp_tensor_transpose'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

