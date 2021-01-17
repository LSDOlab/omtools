from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import omtools.examples.ex_cross as example


vec1 = np.arange(3)
vec2 = np.arange(3) + 1

shape = (2,5,4,3)
num_elements = np.prod(shape)

ten1 = np.arange(num_elements).reshape(shape)
ten2 = np.arange(num_elements).reshape(shape) + 6


def test_vecvec_cross():

    desired_output = np.cross(vec1, vec2)
    np.testing.assert_almost_equal(example.prob['VecVecCross'], desired_output) 

    partials_error = example.prob.check_partials(includes=['comp_VecVecCross'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_tenten_cross():

    desired_output = np.cross(ten1, ten2, axis=3)
    np.testing.assert_almost_equal(example.prob['TenTenCross'], desired_output) 
    
    partials_error = example.prob.check_partials(includes=['comp_TenTenCross'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)


