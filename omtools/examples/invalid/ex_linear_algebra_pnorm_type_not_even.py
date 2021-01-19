from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ErrorPnormTypeNotEven(Group):
    def setup(self):

        # Shape of the tensor
        shape = (2, 3, 4, 5)

        # Number of elements in the tensor
        num_of_elements = np.prod(shape)

        # Creating a numpy tensor with the desired shape and size
        tensor = np.arange(num_of_elements).reshape(shape)

        # Declaring in1 as input tensor
        in1 = self.declare_input('in1', val=tensor)

        # Computing the 6-norm on in1 without specifying an axis
        self.register_output('axis_free_pnorm', ot.pnorm(in1, pnorm_type=3))


prob = Problem()
prob.model = ErrorPnormTypeNotEven()
prob.setup(force_alloc_complex=True)
prob.run_model()
