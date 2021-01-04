from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import omtools.examples.ex_matmat as example

m = 2
n = 3
o = 4
p = 5
q = 6

tensor_shape = (m,n,o,p,q)
num_of_elements = np.prod(tensor_shape)
tensor = np.arange(num_of_elements).reshape(tensor_shape)

def test_axiswise_min():

    desired_output = np.amin(tensor, axis=(1,3))
    np.testing.assert_almost_equal(example.prob['AxiswiseMin'], desired_output) 

    partials_error = example.prob.check_partials(includes=['comp_AxiswiseMin'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

def test_scalar_min():

    desired_output = np.min(tensor)
    np.testing.assert_almost_equal(example.prob['ScalarMin'], desired_output) 
    
    partials_error = example.prob.check_partials(includes=['comp_ScalarMin'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

def test_elementwise_min():

    mat1 = np.arange(m*n).reshape(shape) * 0.5
    mat2 = np.arange(m*n).reshape(shape) * -1.
    mat3 = np.arange(m*n).reshape(shape) 

    desired_output = np.min(mat1, mat2, mat3)
    np.testing.assert_almost_equal(example.prob['ElementwiseMin'], desired_output) 
    
    partials_error = example.prob.check_partials(includes=['comp_ElementwiseMin'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)
