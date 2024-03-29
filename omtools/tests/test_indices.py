import numpy as np
import pytest
from openmdao.utils.assert_utils import assert_check_partials


def test_integer_index_assignement_overlap():
    with pytest.raises(ValueError):
        import omtools.examples.invalid.ex_indices_integer_overlap as example


def test_integer_index_assignement_out_of_range():
    with pytest.raises(ValueError):
        import omtools.examples.invalid.ex_indices_integer_out_of_range as example


def test_integer_index_assignement():
    import omtools.examples.valid.ex_indices_integer as example
    x = np.array([0, 1, 2, 7.4, np.pi, 9, np.pi + 9])
    np.testing.assert_array_equal(example.prob['x'], x)
    np.testing.assert_array_equal(example.prob['x0'], x[0])
    np.testing.assert_array_equal(example.prob['x6'], x[6])
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_integer_index_integer_reuse():
    with pytest.raises(KeyError):
        import omtools.examples.invalid.ex_indices_integer_reuse as example


def test_one_dimensional_index_reuse():
    with pytest.raises(KeyError):
        import omtools.examples.invalid.ex_indices_one_dimensional_reuse as example


def test_one_dimensional_index_assignement_overlap():
    with pytest.raises(ValueError):
        import omtools.examples.invalid.ex_indices_one_dimensional_overlap as example


def test_one_dimensional_index_assignement_out_of_range():
    with pytest.raises(ValueError):
        import omtools.examples.invalid.ex_indices_one_dimensional_out_of_range as example


def test_one_dimensional_index_assignement():
    import omtools.examples.valid.ex_indices_one_dimensional as example
    np.testing.assert_array_equal(example.prob['u'], np.arange(20))
    np.testing.assert_array_equal(example.prob['v'], np.arange(16))
    np.testing.assert_array_equal(example.prob['w'], 16 + np.arange(4))
    x = 2 * (np.arange(20) + 1)
    np.testing.assert_array_equal(example.prob['x'], x)
    np.testing.assert_array_equal(example.prob['x0_5'], x[0:5])
    np.testing.assert_array_equal(example.prob['x3_'], x[3:])
    np.testing.assert_array_equal(example.prob['x2_4'], x[2:4])
    np.testing.assert_array_equal(example.prob['z'], x[2])

    y = np.zeros(20)
    y[:16] = 2 * (np.arange(16) + 1)
    y[16:] = 16 + np.arange(4) - 3
    np.testing.assert_array_equal(example.prob['y'], y)
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)


def test_multidimensional_dimensional_index_assignement_overlap():
    with pytest.raises(KeyError):
        import omtools.examples.invalid.ex_indices_multidimensional_overlap as example


def test_multidimensional_index_assignement_out_of_range():
    with pytest.raises(ValueError):
        import omtools.examples.invalid.ex_indices_multidimensional_out_of_range as example


def test_multidimensional_dimensional_index_assignement():
    import omtools.examples.valid.ex_indices_multidimensional as example
    np.testing.assert_array_equal(
        example.prob['z'],
        np.arange(6).reshape((2, 3)),
    )
    np.testing.assert_array_equal(
        example.prob['x'],
        np.arange(6).reshape((2, 3)),
    )
    p = np.arange(30).reshape((5, 2, 3))
    np.testing.assert_array_equal(example.prob['p'], p)
    # TODO: numpy convention causes shape error without reshape
    np.testing.assert_array_equal(example.prob['r'], p[0, :, :].reshape(
        (1, 2, 3)))
    np.testing.assert_array_equal(
        example.prob['q'],
        np.arange(30).reshape((5, 2, 3)),
    )
    u = np.arange(20).reshape((1, 20))
    v = 2 * np.arange(20).reshape((1, 20))
    w = np.zeros(40).reshape((2, 20))
    w[0, :] = u
    w[1, :] = v
    np.testing.assert_array_equal(
        example.prob['s'],
        w,
    )

    t = np.ones(np.prod((5, 3, 3))).reshape((5, 3, 3))
    t[0:5, 0:-1, 0:3] = np.arange(30).reshape((5, 2, 3))
    np.testing.assert_array_equal(
        example.prob['t'],
        t,
    )
    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)
