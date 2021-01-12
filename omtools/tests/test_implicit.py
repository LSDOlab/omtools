from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest

# def test_implicit_nonlinearc():
#     import omtools.examples.valid.ex_implicit_apply_nonlinear as example
#     np.testing.assert_almost_equal(example.x1, np.array([1.0]))
#     np.testing.assert_almost_equal(example.x2, np.array([3.0]))


def test_solve_quadratic_bracketed_scalar():
    import omtools.examples.valid.ex_implicit_bracketed_scalar as example
    np.testing.assert_almost_equal(example.prob['x'], np.array([1.0]))


def test_solve_quadratic_bracketed_array():
    import omtools.examples.valid.ex_implicit_bracketed_array as example
    np.testing.assert_almost_equal(
        example.prob['x'],
        np.array([1.0, 3.0]),
    )


# def test_implicit_nonlinear_with_subsystems_in_residual():
#     import omtools.examples.valid. as example
#     np.testing.assert_almost_equal(example.prob['y'], np.array([1.07440944]))


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
