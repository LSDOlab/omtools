from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest


def test_integer_index_assignement_overlap():
    with pytest.raises(ValueError):
        import omtools.examples.ex_integer_index_assignment_overlap as example


def test_integer_index_assignement_out_of_range():
    with pytest.raises(ValueError):
        import omtools.examples.ex_integer_index_assignment_out_of_range as example


def test_integer_index_assignement():
    import omtools.examples.ex_integer_index_assignment as example
    np.testing.assert_array_equal(
        example.prob['x'], np.array([
            0,
            1,
            2,
            7.4,
            np.pi,
            9,
            np.pi + 9,
        ], ))
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_one_dimensional_index_assignement_overlap():
    with pytest.raises(ValueError):
        import omtools.examples.ex_one_dimensional_index_assignment_overlap as example


def test_one_dimensional_index_assignement_out_of_range():
    with pytest.raises(ValueError):
        import omtools.examples.ex_one_dimensional_index_assignment_out_of_range as example


def test_one_dimensional_index_assignement():
    import omtools.examples.ex_one_dimensional_index_assignment as example
    np.testing.assert_array_equal(example.prob['u'], np.arange(20))
    np.testing.assert_array_equal(example.prob['v'], np.arange(16))
    np.testing.assert_array_equal(example.prob['w'], 16 + np.arange(4))
    np.testing.assert_array_equal(example.prob['x'], 2 * (np.arange(20) + 1))
    y = np.zeros(20)
    y[:16] = 2 * (np.arange(16) + 1)
    y[16:] = 16 + np.arange(4) - 3
    np.testing.assert_array_equal(example.prob['y'], y)
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_multidimensional_dimensional_index_assignement_overlap():
    with pytest.raises(ValueError):
        import omtools.examples.ex_multidimensional_index_assignment_overlap as example


def test_multidimensional_index_assignement_out_of_range():
    with pytest.raises(ValueError):
        import omtools.examples.ex_multidimensional_index_assignment_out_of_range as example


def test_multidimensional_dimensional_index_assignement():
    import omtools.examples.ex_multidimensional_index_assignment as example
    np.testing.assert_array_equal(
        example.prob['z'],
        np.arange(6).reshape((2, 3)),
    )
    np.testing.assert_array_equal(
        example.prob['x'],
        np.arange(6).reshape((2, 3)),
    )
    np.testing.assert_array_equal(
        example.prob['p'],
        np.arange(30).reshape((5, 2, 3)),
    )
    np.testing.assert_array_equal(
        example.prob['q'],
        np.arange(30).reshape((5, 2, 3)),
    )
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)
