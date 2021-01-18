from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_literals():
    import omtools.examples.valid.ex_explicit_literals as example
    np.testing.assert_approx_equal(example.prob['y'], -3.)
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_simple_binary():
    import omtools.examples.valid.ex_explicit_binary_operations as example
    np.testing.assert_approx_equal(example.prob['y1'], 5.)
    np.testing.assert_approx_equal(example.prob['y2'], 1.)
    np.testing.assert_approx_equal(example.prob['y3'], 6.)
    np.testing.assert_approx_equal(example.prob['y4'], 2 / 3.)
    np.testing.assert_approx_equal(example.prob['y5'], 2 / 3.)
    np.testing.assert_approx_equal(example.prob['y6'], 2 / 3.)
    np.testing.assert_approx_equal(example.prob['y7'], 4.)
    np.testing.assert_approx_equal(example.prob['y8'], 9.)
    np.testing.assert_approx_equal(example.prob['y9'], 14.)
    np.testing.assert_array_almost_equal(example.prob['y10'], np.arange(7)**2)
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_no_registered_outputs():
    import omtools.examples.valid.ex_explicit_no_registered_output as example
    np.testing.assert_approx_equal(example.prob['prod'], 24.)
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)
    assert len(example.prob.model._subgroups_myproc) == 1


def test_unary_exprs():
    import omtools.examples.valid.ex_explicit_unary as example
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


def test_explicit_with_subsystems():
    import omtools.examples.valid.ex_explicit_with_subsystems as example
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


def test_explicit_cycles():
    import omtools.examples.valid.ex_explicit_cycles as example
    np.testing.assert_approx_equal(
        example.prob['cycle_1.x'],
        1.1241230297043157,
    )
    np.testing.assert_approx_equal(
        example.prob['cycle_2.x'],
        1.0798960718178603,
    )
    np.testing.assert_almost_equal(example.prob['cycle_3.x'], 0.)
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    # assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_unary_exprs():
    import omtools.examples.valid.ex_explicit_unary as example
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
