from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleMatVecProduct(Group):
    def setup(self):
        m = 3
        n = 2
        p = 4

        # Shape of the first matrix (3,2)
        shape1 = (m, n)

        # Shape of the second matrix (2,4)
        shape2 = (n, p)

        # Creating the values of both matrices
        val1 = np.arange(m * n).reshape(shape1)

        # Creating the values for the vector
        val3 = np.arange(n)

        # Declaring the two input matrices as mat1 and mat2
        mat1 = self.declare_input('mat1', val=val1)

        # Declaring the input vector of size (n,)
        vec1 = self.declare_input('vec1', val=val3)

        # Creating the output for a matrix multiplied by a vector
        self.register_output('MatVec', ot.matmat(mat1, vec1))


prob = Problem()
prob.model = ExampleMatVecProduct()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('mat1', prob['mat1'].shape)
print(prob['mat1'])
print('vec1', prob['vec1'].shape)
print(prob['vec1'])
print('MatVec', prob['MatVec'].shape)
print(prob['MatVec'])
