from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleTensorTensor(Group):
    def setup(self):

        m = 3
        n = 4
        p = 5

        # Shape of the tensors
        ten_shape = (m, n, p)

        # Number of elements in the tensors
        num_ten_elements = np.prod(ten_shape)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements).reshape(ten_shape)
        ten2 = np.arange(num_ten_elements,
                         2 * num_ten_elements).reshape(ten_shape)

        # Adding tensors to omtools
        ten1 = self.declare_input('ten1', val=ten1)
        ten2 = self.declare_input('ten2', val=ten2)

        # Tensor-Tensor Inner Product specifying the first and last axes
        self.register_output(
            'TenTenInner',
            ot.inner(ten1, ten2, axes=([0, 2], [0, 2])),
        )


prob = Problem()
prob.model = ExampleTensorTensor()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('ten1', prob['ten1'].shape)
print(prob['ten1'])
print('ten2', prob['ten2'].shape)
print(prob['ten2'])
print('TenTenInner', prob['TenTenInner'].shape)
print(prob['TenTenInner'])
