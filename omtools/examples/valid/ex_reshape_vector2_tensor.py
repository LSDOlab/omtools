from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleVector2Tensor(Group):
    def setup(self):
        shape = (2, 3, 4, 5)
        size = 2 * 3 * 4 * 5

        # Both vector or tensors need to be numpy arrays
        tensor = np.arange(size).reshape(shape)
        vector = np.arange(size)

        # in2 is a vector having a size of 2 * 3 * 4 * 5
        in2 = self.declare_input('in2', val=vector)

        # in2 is being reshaped from size =  2 * 3 * 4 * 5 to a tenßsor
        # having shape = (2, 3, 4, 5)
        self.register_output('reshape_vector2tensor',
                             ot.reshape(in2, new_shape=shape))


prob = Problem()
prob.model = ExampleVector2Tensor()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('in2', prob['in2'].shape)
print(prob['in2'])
print('reshape_vector2tensor', prob['reshape_vector2tensor'].shape)
print(prob['reshape_vector2tensor'])
