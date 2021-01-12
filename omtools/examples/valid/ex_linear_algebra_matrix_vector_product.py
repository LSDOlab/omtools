from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleMatrixVectorProduct(Group):
    def setup(self):
        m = 3
        n = 4

        # Shape of the first matrix (3,2)
        shape1 = (m, n)

        # Shape of the second matrix (2,4)
        shape2 = (n, )

        # Creating the values of both matrices
        val1 = np.arange(m * n).reshape(shape1)
        val2 = np.arange(n).reshape(shape2)

        # Declaring the input matrix and input vector
        mat1 = self.declare_input('mat1', val=val1)
        vec1 = self.declare_input('vec1', val=val2)

        # Creating the output for matrix-vector multiplication
        self.register_output('MatVec', ot.matvec(mat1, vec1))


prob = Problem()
prob.model = ExampleMatrixVectorProduct()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('MatVec', prob['MatVec'].shape)
print(prob['MatVec'])
