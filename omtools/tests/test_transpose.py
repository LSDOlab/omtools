from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_matrix_transpose():
    import omtools.examples.valid.ex_transpose_matrix as example

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


def test_tensor_transpose():
    import omtools.examples.valid.ex_transpose_tensor as example

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