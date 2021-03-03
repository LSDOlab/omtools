from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleAxisWise(Group):
    def setup(self):

        # Shape of the tensor
        shape = (2, 3, 4, 5)

        # Number of elements in the tensor
        num_of_elements = np.prod(shape)

        # Creating a numpy tensor with the desired shape and size
        tensor = np.arange(num_of_elements).reshape(shape)

        # Declaring in1 as input tensor
        in1 = self.declare_input('in1', val=tensor)

        # Computing the 6-norm of in1 over the specified axes.
        self.register_output('axiswise_pnorm',
                             ot.pnorm(in1, axis=(1, 3), pnorm_type=6))


prob = Problem()
prob.model = ExampleAxisWise()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('in1', prob['in1'].shape)
print(prob['in1'])
print('axiswise_pnorm', prob['axiswise_pnorm'].shape)
print(prob['axiswise_pnorm'])
