from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group

class Example(Group):

    def setup(self):

        # Declare mat as an input matrix with shape = (4, 2)
        mat = self.declare_input('M1', val=np.arange(4 * 2).reshape((4, 2)))  
 
        # Declare tens as an input tensor with shape = (4, 3, 2, 5)
        tens = self.declare_input('T1', val=np.arange(4 * 3 * 5 * 2).reshape((4, 3, 5, 2)))  

        # Compute the transpose of mat
        self.register_output('axes_reordered_matrix', ot.reorder_axes(mat, 'ij->ji'))

        # Compute a new tensor by reordering axes of tens; reordering is given by 'ijkl->ljki'
        self.register_output('axes_reordered_tensor', ot.reorder_axes(tens, 'ijkl->ljki'))


        

prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.run_model()

prob.check_partials(compact_print=True)