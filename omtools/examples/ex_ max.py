from openmdao.api import NonlinearBlockGS, ScipyKrylov, NewtonSolver
from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group
from omtools.core.expression import Expression


class Example(Group):
    def setup(self):
        m = 2
        n = 3
        o = 4
        p = 5
        q = 6

        """
        Scalar and Axis-wise maximum of a tensor across the (1,3) axis
        """

        # Shape of a tensor 
        tensor_shape = (m,n,o,p,q)

        num_of_elements = np.prod(tensor_shape)
        # Creating the values of the tensor
        val = np.arange(num_of_elements).reshape(tensor_shape)

        # Declaring the tensor as an input 
        ten1 = self.declare_input('ten1', val=val)        

        # Computing the axiswise minimum on the tensor
        axis = (1,3)
        self.register_output('AxiswiseMax', ot.max(ten1, axis=axis))

        # Computing the minimum across the entire tensor, returns single value
        self.register_output('ScalarMax', ot.max(ten1))

        """
        Element-wise minimum between three tensors 
        """
        m = 2
        n = 3
        # Shape of the three matrices is (2,3)
        shape = (m,n)

        # Creating the values for all three matrices 
        val1 = np.arange(m*n).reshape(shape) * 0.5
        val2 = np.arange(m*n).reshape(shape) * -1.
        val3 = np.arange(m*n).reshape(shape) 

        # Declaring the three input matrices
        mat1 = self.declare_input('mat1', val=val1)
        mat2 = self.declare_input('mat2', val=val2)     
        mat3 = self.declare_input('mat3', val=val3)

        # Creating the output for matrix multiplication
        self.register_output('ElementwiseMax', ot.max(mat1, mat2, mat3))


        

prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.check_partials(compact_print=True)
prob.run_model()

prob.model.list_inputs(prom_name=True, print_arrays=True)
prob.model.list_outputs(prom_name=True, print_arrays=True)