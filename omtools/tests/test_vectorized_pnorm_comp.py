from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import omtools.examples.ex_pnorm_comp as example

i = 2
j = 3
k = 4
l = 5
shape = (i, j, k, l)
pnorm_type = 6

val = np.arange(np.prod(shape)).reshape(shape)

def test_pnorm_axisfree():

    desired_output = np.linalg.norm(val.flatten(), ord=pnorm_type)
    np.testing.assert_almost_equal(example.prob['axis_free_pnorm'], desired_output) 

    partials_error = example.prob.check_partials(includes=['comp_axis_free_pnorm'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)


def test_pnorm_axiswise():

    axis  = (1,3)
    desired_output = np.sum(val**pnorm_type, axis=axis) ** (1/pnorm_type)

    np.testing.assert_almost_equal(example.prob['axiswise_pnorm'], desired_output) 

    partials_error = example.prob.check_partials(includes=['comp_axiswise_pnorm'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)




# def test_vector_norm():
#     val = np.arange(10)
#     desired_output = np.linalg.norm(val, 2)

#     np.testing.assert_almost_equal(example.prob['vector_norm_2'], desired_output) 


#     partials_error = example.prob.check_partials(includes=['comp_vector_norm_2'], out_stream=None, compact_print=True)
#     assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


    
# def test_matrix_norm():
#     n = 10
#     m = 20
#     val = np.arange(n* m).reshape((n, m))
    
#     desired_output = np.linalg.norm(val, 'fro')

#     np.testing.assert_almost_equal(example.prob['matrix_norm_fro'], desired_output) 


#     partials_error = example.prob.check_partials(includes=['comp_matrix_norm_fro'], out_stream=None, compact_print=True)
#     assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)

# def test_matrix_axis_norm():
#     n = 10
#     m = 20
#     val = np.arange(n* m).reshape((n, m))
    
#     desired_output_0 = np.linalg.norm(val, 2, axis=0)
#     desired_output_1 = np.linalg.norm(val, 2, axis=1)
        

#     np.testing.assert_array_almost_equal(example.prob['axis_0_norm_2'], desired_output_0) 
#     np.testing.assert_array_almost_equal(example.prob['axis_1_norm_2'], desired_output_1)

#     partials_error0 = example.prob.check_partials(includes=['comp_axis_0_norm_2'],out_stream=None, compact_print=True)
#     partials_error1 = example.prob.check_partials(includes=['comp_axis_1_norm_2'],out_stream=None, compact_print=True)

#     assert_check_partials(partials_error0, atol=1.e-5, rtol=1.e-5)
#     assert_check_partials(partials_error1, atol=1.e-5, rtol=1.e-5)
    


