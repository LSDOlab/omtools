from openmdao.api import NonlinearBlockGS, ScipyKrylov, NewtonSolver
from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group
from omtools.core.expression import Expression


class Example(Group):
    def setup(self):

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

        # Adding the vectors and tensors to omtools
        vec1 = self.declare_input('vec1', val=vec1)
        vec2 = self.declare_input('vec2', val=vec2)

        ten1 = self.declare_input('ten1', val=ten1)      
        ten2 = self.declare_input('ten2', val=ten2)
        
        # Vector-Vector Inner Product
        self.register_output('VecVecInner', ot.inner(vec1, vec2))

        # Tensor-Vector Inner Product specifying the first axis for Vector and Tensor
        self.register_output('TenVecInner', ot.inner(ten1, vec1, axes=([0],[0])) )

        # Tensor-Tensor Inner Product specifying the first and last axes
        self.register_output('TenTenInner', ot.inner(ten1, ten2, axes=([0,2],[0,2])) )

        

prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.check_partials(compact_print=True)
prob.run_model()
