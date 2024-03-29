from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_pnorm_axisfree_norm():
    import omtools.examples.valid.ex_pnorm_axis_free as example

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


def test_pnorm_axiswise():
    import omtools.examples.valid.ex_pnorm_axis_wise as example

    i = 2
    j = 3
    k = 4
    l = 5
    shape = (i, j, k, l)
    pnorm_type = 6

    val = np.arange(np.prod(shape)).reshape(shape)

    axis = (1, 3)
    desired_output = np.sum(val**pnorm_type, axis=axis)**(1 / pnorm_type)

    np.testing.assert_almost_equal(example.prob['axiswise_pnorm'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_axiswise_pnorm'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)


def test_pnorm_type_not_positive():
    with pytest.raises(Exception):
        import omtools.examples.invalid.ex_pnorm_type_not_positive as example


def test_pnorm_type_not_even():
    with pytest.raises(Exception):
        import omtools.examples.invalid.ex_pnorm_type_not_even as example
