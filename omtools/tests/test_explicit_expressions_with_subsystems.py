from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_simple_binary_with_subsystem():
    import omtools.examples.ex_simple_explicit_with_subsystems as example
    np.testing.assert_approx_equal(example.prob['x1'], 40.)
    np.testing.assert_approx_equal(example.prob['x2'], 12.)
    np.testing.assert_approx_equal(example.prob['y1'], 52.)
    np.testing.assert_approx_equal(example.prob['y2'], -28.)
    np.testing.assert_approx_equal(example.prob['y3'], 480.)
    np.testing.assert_approx_equal(example.prob['prod'], 480.)
    np.testing.assert_approx_equal(example.prob['y4'], 1600.)
    np.testing.assert_approx_equal(example.prob['y5'], 144.)
    np.testing.assert_approx_equal(example.prob['y6'], 196.)
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_no_registered_outputs():
    import omtools.examples.ex_no_registered_outputs as example
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)
    assert len(example.prob.model._subgroups_myproc) == 1
