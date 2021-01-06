from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import omtools.examples.ex_matmat as example

m = 3
n = 2
p = 4

# Shape of the first matrix (3,2)
shape1 = (m,n)

# Shape of the second matrix (2,4)
shape2 = (n,p)

# Creating the values of both matrices 
mat1 = np.arange(m*n).reshape(shape1)
mat2 = np.arange(n*p).reshape(shape2)

# Creating the values for the vector
vec1 = np.arange(n)

def test_matmat():

    desired_output = np.matmul(mat1, mat2)
    np.testing.assert_almost_equal(example.prob['MatMat'], desired_output) 

    partials_error = example.prob.check_partials(includes=['comp_MatMat'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_matvec():

    desired_output = np.matmul(mat1, vec1) 
    np.testing.assert_almost_equal(example.prob['MatVec'], desired_output) 

    partials_error = example.prob.check_partials(includes=['comp_MatVec'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

