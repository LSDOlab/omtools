from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleTensorTensorLast(Group):
    def setup(self):

        m = 2
        n = 4
        p = 3

        # Shape of the tensors
        ten_shape = (m, n, p)

        # Number of elements in the tensors
        num_ten_elements = np.prod(ten_shape)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements).reshape(ten_shape)
        ten2 = np.arange(num_ten_elements,
                         2 * num_ten_elements).reshape(ten_shape)

        # Adding the tensors to omtools
        ten1 = self.declare_input('ten1', val=ten1)
        ten2 = self.declare_input('ten2', val=ten2)

        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('TenTenDotLast', ot.dot(ten1, ten2, axis=2))


prob = Problem()
prob.model = ExampleTensorTensorLast()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('ten1', prob['ten1'].shape)
print(prob['ten1'])
print('ten2', prob['ten2'].shape)
print(prob['ten2'])
print('TenTenDotLast', prob['TenTenDotLast'].shape)
print(prob['TenTenDotLast'])
