from openmdao.utils.assert_utils import assert_check_partials
import numpy as np

def test_outer_product():
    import omtools.examples.ex_einsum_outer_product as example1
    import omtools.examples.ex_einsum_outer_product_sparse_derivs as example2
    import omtools.examples.ex_einsum_outer_product_new_api as example3

    desired_output = np.einsum('i,j->ij', np.arange(5), np.arange(4))

    np.testing.assert_array_almost_equal(example1.prob['f'], desired_output) 
    np.testing.assert_array_almost_equal(example2.prob['f'], desired_output)  
    np.testing.assert_array_almost_equal(example3.prob['f'], desired_output)
    
    partials_error1 = example1.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(partials_error1, atol=1.e-6, rtol=1.e-6)
    
    partials_error2 = example2.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(partials_error2, atol=1.e-6, rtol=1.e-6)
    
    partials_error3 = example3.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(partials_error3, atol=1.e-6, rtol=1.e-6)
    

def test_inner_product():
    import omtools.examples.ex_einsum_inner_product as example1
    # import omtools.examples.ex_einsum_inner_product_sparse_derivs as example2
    import omtools.examples.ex_einsum_inner_product_new_api as example3

    desired_output = np.einsum('ij,j->i', np.arange(20).reshape((5, 4)), np.arange(4))

    np.testing.assert_array_almost_equal(example1.prob['f'], desired_output) 
    # np.testing.assert_array_almost_equal(example2.prob['f'], desired_output)  
    np.testing.assert_array_almost_equal(example3.prob['f'], desired_output)
    
    partials_error1 = example1.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(partials_error1, atol=1.e-6, rtol=1.e-6)
    
    # partials_error2 = example2.prob.check_partials(out_stream=None, compact_print=True)
    # assert_check_partials(partials_error2, atol=1.e-6, rtol=1.e-6)
    
    partials_error3 = example3.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(partials_error3, atol=1.e-6, rtol=1.e-6)

