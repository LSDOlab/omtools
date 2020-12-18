from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_simple_binary_with_subsystem():
    import omtools.examples.ex_explicit_cycles as example
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
