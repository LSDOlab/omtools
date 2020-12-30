from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_reuse_residual_error():
    with pytest.raises(ValueError):
        import omtools.examples.ex_reuse_residual_error as examples


def test_implicit_nonlinearc():
    import omtools.examples.ex_implicit_nonlinear as example
    np.testing.assert_almost_equal(example.x1, np.array([1.0]))
    np.testing.assert_almost_equal(example.x2, np.array([3.0]))


def test_solve_quadratic_bracketed_scalar():
    import omtools.examples.ex_implicit_bracketed_scalar as example
    np.testing.assert_almost_equal(example.prob['x'], np.array([1.0]))


def test_solve_quadratic_bracketed_array():
    import omtools.examples.ex_implicit_bracketed_array as example
    np.testing.assert_almost_equal(example.prob['x'], np.array([1.0, 3.0]))
