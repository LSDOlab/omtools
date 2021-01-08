from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import omtools.examples.ex_reshape as example

i = 2
j = 3
k = 4
l = 5
shape = (i, j, k, l)

tensor = np.arange(np.prod(shape)).reshape(shape)
vector = np.arange(np.prod(shape))

def test_tensor2vector():

    desired_output = vector
    np.testing.assert_almost_equal(example.prob['reshape_tensor2vector'], desired_output) 

    partials_error = example.prob.check_partials(includes=['comp_reshape_tensor2vector'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_vector2tensor():

    desired_output = tensor

    np.testing.assert_almost_equal(example.prob['reshape_vector2tensor'], desired_output) 

    partials_error = example.prob.check_partials(includes=['comp_reshape_vector2tensor'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)
