from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import omtools.examples.ex_matvec as example

m = 3
n = 4

# Shape of the first matrix (3,2)
shape1 = (m,n)

# Shape of the vector (4,)
shape2 = (n,)

# Creating the matrix
mat1 = np.arange(m*n).reshape(shape1)

# Creating the vector
vec1 = np.arange(n).reshape(shape2)


def test_matvec():

    desired_output = np.matmul(mat1, vec1)
    np.testing.assert_almost_equal(example.prob['MatVec'], desired_output) 

    partials_error = example.prob.check_partials(includes=['comp_MatVec'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)

