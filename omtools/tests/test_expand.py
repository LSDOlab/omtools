from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import pytest
from openmdao.api import Problem


def test_expand_errors():
    import omtools.examples.ex_expand_errors as example

    with pytest.raises(TypeError):
        prob = Problem()
        prob.model = example.ErrorScalarIncorrectOrder()
        prob.setup()

    with pytest.raises(ValueError):
        prob = Problem()
        prob.model = example.ErrorScalarIndices()
        prob.setup()

    with pytest.raises(ValueError):
        prob = Problem()
        prob.model = example.ErrorArrayNoIndices()
        prob.setup()

    with pytest.raises(ValueError):
        prob = Problem()
        prob.model = example.ErrorArrayInvalidIndices1()
        prob.setup()

    with pytest.raises(ValueError):
        prob = Problem()
        prob.model = example.ErrorArrayInvalidIndices2()
        prob.setup()


def test_expand():
    import omtools.examples.ex_expand as example
    np.testing.assert_array_equal(
        example.prob['scalar'], np.array([
            1
        ])
    )
    np.testing.assert_array_equal(
        example.prob['expanded_scalar'], np.array([
            [1., 1., 1.,],
            [1., 1., 1.,],
        ])
    )

    array = np.array([
        [1., 2., 3.],
        [4., 5., 6.],
    ])
    expanded_array = np.empty((2, 4, 3, 1))
    for i in range(4):
        for j in range(1):
            expanded_array[:, i, :, j] = array

    np.testing.assert_array_equal(
        example.prob['array'], array
    )
    np.testing.assert_array_equal(
        example.prob['expanded_array'], expanded_array
    )

    result = example.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(result, atol=1.e-8, rtol=1.e-8)