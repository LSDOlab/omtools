from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleTensorVector(Group):
    def setup(self):

        m = 3
        n = 4
        p = 5

        # Shape of the vectors
        vec_shape = (m, )

        # Shape of the tensors
        ten_shape = (m, n, p)

        # Values for the two vectors
        vec1 = np.arange(m)

        # Number of elements in the tensors
        num_ten_elements = np.prod(ten_shape)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements).reshape(ten_shape)

        # Adding the vector and tensor to omtools
        vec1 = self.declare_input('vec1', val=vec1)

        ten1 = self.declare_input('ten1', val=ten1)

        # Tensor-Vector Outer Product specifying the first axis for Vector and Tensor
        self.register_output('TenVecOuter', ot.outer(ten1, vec1))


prob = Problem()
prob.model = ExampleTensorVector()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('vec1', prob['vec1'].shape)
print(prob['vec1'])
print('ten1', prob['ten1'].shape)
print(prob['ten1'])
print('TenVecOuter', prob['TenVecOuter'].shape)
print(prob['TenVecOuter'])
