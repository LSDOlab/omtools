from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import omtools.examples.ex_dot as example

m = 3
n = 4
p = 5

# Shape of the vectors
vec_shape = (m,)

# Shape of the tensors
ten_shape = (m,n,p)

# Values for the two vectors
vec1 = np.arange(m)
vec2 = np.arange(m, 2*m)

# Number of elements in the tensors
num_ten_elements = np.prod(ten_shape)

# Values for the two tensors
ten1 = np.arange(num_ten_elements).reshape(ten_shape)
ten2 = np.arange(num_ten_elements, 2*num_ten_elements).reshape(ten_shape)


def test_vecvec_dot():

    desired_output = np.dot(vec1, vec2)
    np.testing.assert_almost_equal(example.prob['VecVecDot'], desired_output) 

    partials_error = example.prob.check_partials(includes=['comp_VecVecDot'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_tenten_dot():

    desired_output = np.tensordot(ten1, ten2, axes=([0],[0]))
    np.testing.assert_almost_equal(example.prob['TenTenDotFirst'], desired_output) 
    
    partials_error = example.prob.check_partials(includes=['comp_TenTenDotFirst'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)


    desired_output = np.tensordot(ten1, ten2, axes=([2],[2]))
    np.testing.assert_almost_equal(example.prob['TenTenDotLast'], desired_output) 
    
    partials_error = example.prob.check_partials(includes=['comp_TenTenDotLast'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-4, rtol=1.e-4)

