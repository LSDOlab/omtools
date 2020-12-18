from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_registering_input_causes_error():
    with pytest.raises(TypeError):
        import omtools.examples.ex_register_input_fails


def test_unused_inputs_create_no_subsystems():
    from openmdao.api import Group
    import omtools.examples.ex_unused_inputs as example
    assert example.prob.model._group_inputs == {}
    assert example.prob.model._subsystems_allprocs == []


def test_indep_var():
    import omtools.examples.ex_create_indep_var as example
    np.testing.assert_approx_equal(example.prob['z'], 10.)
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_literals():
    import omtools.examples.ex_working_with_literal_values as example
    np.testing.assert_approx_equal(example.prob['y'], -3.)
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_simple_binary():
    import omtools.examples.ex_simple_binary_operations as example
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


# options
