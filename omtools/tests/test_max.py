from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_max_scalar():
    import omtools.examples.valid.ex_max_scalar as example

    m = 2
    n = 3
    o = 4
    p = 5
    q = 6

    tensor_shape = (m, n, o, p, q)
    num_of_elements = np.prod(tensor_shape)
    tensor = np.arange(num_of_elements).reshape(tensor_shape)

    # SCALAR MIN
    desired_output = np.max(tensor)
    np.testing.assert_almost_equal(example.prob['ScalarMin'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_ScalarMin'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_max_axiswise():
    import omtools.examples.valid.ex_max_axiswise as example

    m = 2
    n = 3
    o = 4
    p = 5
    q = 6

    tensor_shape = (m, n, o, p, q)
    num_of_elements = np.prod(tensor_shape)
    tensor = np.arange(num_of_elements).reshape(tensor_shape)

    # AXISWISE MIN
    desired_output = np.amax(tensor, axis=1)
    np.testing.assert_almost_equal(example.prob['AxiswiseMin'], desired_output)

    partials_error = example.prob.check_partials(includes=['comp_AxiswiseMin'],
                                                 out_stream=None,
                                                 compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_max_elementwise():
    import omtools.examples.valid.ex_max_elementwise as example

    tensor1 = np.array([[1, 5, -8], [10, -3, -5]])
    tensor2 = np.array([[2, 6, 9], [-1, 2, 4]])

    desired_output = np.maximum(tensor1, tensor2)
    np.testing.assert_almost_equal(example.prob['ElementwiseMin'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_ElementwiseMin'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_max_multi_inputs_and_axis():
    with pytest.raises(Exception):
        import omtools.examples.invalid.ex_max_multi_inputs_and_axis as example


def test_max_inputs_not_same_size():
    with pytest.raises(Exception):
        import omtools.examples.invalid.ex_max_inputs_not_same_size as example
