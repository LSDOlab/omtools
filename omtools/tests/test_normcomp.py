from openmdao.utils.assert_utils import assert_check_partials
import numpy as np

def test_vector_norm():
    import omtools.examples.ex_vector_norm_2 as example1

    val = np.arange(10)
    desired_output = np.linalg.norm(val, 2)

    np.testing.assert_almost_equal(example1.prob['y'], desired_output) 


    partials_error1 = example1.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(partials_error1, atol=1.e-6, rtol=1.e-6)
    
def test_matrix_norm():
    import omtools.examples.ex_matrix_norm_fro as example1
    n = 10
    m = 20
    val = np.arange(n* m).reshape((n, m))
    
    desired_output = np.linalg.norm(val, 'fro')

    np.testing.assert_almost_equal(example1.prob['y'], desired_output) 


    partials_error1 = example1.prob.check_partials(out_stream=None, compact_print=True)
    assert_check_partials(partials_error1, atol=1.e-5, rtol=1.e-5)