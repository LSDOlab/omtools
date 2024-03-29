from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_indep_var():
    import omtools.examples.valid.ex_indep_var_simple as example
    np.testing.assert_approx_equal(example.prob['z'], 10.)
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)
