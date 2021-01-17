from openmdao.api import NonlinearBlockGS, ScipyKrylov, NewtonSolver
from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group
from omtools.core.expression import Expression


class Example(Group):
    def setup(self):

        # Creating two vectors 
        vecval1 = np.arange(3)
        vecval2 = np.arange(3) + 1

        vec1 = self.declare_input('vec1', val=vecval1)
        vec2 = self.declare_input('vec2', val=vecval2)

        # Vector-Vector Cross Product
        self.register_output('VecVecCross', ot.cross(vec1, vec2, axis=0))

        # Creating two tensors 
        shape = (2,5,4,3)
        num_elements = np.prod(shape)

        tenval1 = np.arange(num_elements).reshape(shape)
        tenval2 = np.arange(num_elements).reshape(shape) + 6

        ten1 = self.declare_input('ten1', val=tenval1)      
        ten2 = self.declare_input('ten2', val=tenval2)
        
        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('TenTenCross', ot.cross(ten1, ten2, axis=3))
        

prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.check_partials(compact_print=True)
prob.run_model()

prob.model.list_inputs(print_arrays=True)
prob.model.list_outputs(print_arrays=True)
