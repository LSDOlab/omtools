from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_implicit_nonlinear():
    import omtools.examples.valid.ex_implicit_apply_nonlinear as example

    example.prob.set_val('x', 1.9)
    example.prob.run_model()
    np.testing.assert_almost_equal(example.prob['x'], np.array([1.0]))

    example.prob.set_val('x', 2.1)
    example.prob.run_model()
    np.testing.assert_almost_equal(example.prob['x'], np.array([3.0]))


def test_solve_quadratic_bracketed_scalar():
    import omtools.examples.valid.ex_implicit_bracketed_scalar as example
    np.testing.assert_almost_equal(example.prob['x'], np.array([1.0]))


def test_solve_quadratic_bracketed_array():
    import omtools.examples.valid.ex_implicit_bracketed_array as example
    np.testing.assert_almost_equal(
        example.prob['x'],
        np.array([1.0, 3.0]),
    )


def test_implicit_nonlinear_with_subsystems_in_residual():
    import omtools.examples.valid.ex_implicit_with_subsystems as example

    # example.prob.set_val('y', 1.9)
    # example.prob.run_model()
    # print(example.prob['y'])
    np.testing.assert_almost_equal(example.prob['y'], np.array([1.07440944]))


def test_implicit_nonlinear_with_subsystems_bracketed_scalar():
    import omtools.examples.valid.ex_implicit_with_subsystems_bracketed_scalar as example
    np.testing.assert_almost_equal(
        example.prob['y'],
        np.array([1.07440944]),
    )


def test_implicit_nonlinear_with_subsystems_bracketed_array():
    import omtools.examples.valid.ex_implicit_with_subsystems_bracketed_array as example
    np.testing.assert_almost_equal(
        example.prob['y'],
        np.array([1.07440944, 2.48391993]),
    )
