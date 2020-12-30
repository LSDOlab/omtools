from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_solve_quadratic():
    import omtools.examples.ex_implicit_relationships as example
    np.testing.assert_almost_equal(example.x1, 1.0)
    np.testing.assert_almost_equal(example.x2, 3.0)
