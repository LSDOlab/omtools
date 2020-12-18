from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_unary_exprs():
    import omtools.examples.ex_unary_exprs as example
    x = np.pi
    y = 1
    np.testing.assert_approx_equal(example.prob['arccos'], np.arccos(y))
    np.testing.assert_approx_equal(example.prob['arcsin'], np.arcsin(y))
    np.testing.assert_approx_equal(example.prob['arctan'], np.arctan(x))
    np.testing.assert_approx_equal(example.prob['cos'], np.cos(x))
    np.testing.assert_approx_equal(example.prob['cosec'], 1 / np.sin(y))
    np.testing.assert_approx_equal(example.prob['cosech'], 1 / np.sinh(x))
    np.testing.assert_approx_equal(example.prob['cosh'], np.cosh(x))
    np.testing.assert_approx_equal(example.prob['cotan'], 1 / np.tan(y))
    np.testing.assert_approx_equal(example.prob['cotanh'], 1 / np.tanh(x))
    np.testing.assert_approx_equal(example.prob['exp'], np.exp(x))
    np.testing.assert_approx_equal(example.prob['log'], np.log(x))
    np.testing.assert_approx_equal(example.prob['log10'], np.log10(x))
    np.testing.assert_approx_equal(example.prob['sec'], 1 / np.cos(x))
    np.testing.assert_approx_equal(example.prob['sech'], 1 / np.cosh(x))
    np.testing.assert_approx_equal(example.prob['sin'], np.sin(x))
    np.testing.assert_approx_equal(example.prob['sinh'], np.sinh(x))
    np.testing.assert_approx_equal(example.prob['tan'], np.tan(x))
    np.testing.assert_approx_equal(example.prob['tanh'], np.tanh(x))
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    # assert_check_partials(result, atol=1.e-8, rtol=1.e-8)
