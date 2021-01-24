from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleTensor2Vector(Group):
    def setup(self):
        shape = (2, 3, 4, 5)
        size = 2 * 3 * 4 * 5

        # Both vector or tensors need to be numpy arrays
        tensor = np.arange(size).reshape(shape)
        vector = np.arange(size)

        # in1 is a tensor having shape = (2, 3, 4, 5)
        in1 = self.declare_input('in1', val=tensor)

        # in1 is being reshaped from shape = (2, 3, 4, 5) to a vector
        # having size = 2 * 3 * 4 * 5
        self.register_output('reshape_tensor2vector',
                             ot.reshape(in1, new_shape=(size, )))


prob = Problem()
prob.model = ExampleTensor2Vector()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('in1', prob['in1'].shape)
print(prob['in1'])
print('reshape_tensor2vector', prob['reshape_tensor2vector'].shape)
print(prob['reshape_tensor2vector'])
