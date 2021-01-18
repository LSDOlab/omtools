from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleDot(Group):
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
        vec2 = np.arange(m, 2 * m)

        # Number of elements in the tensors
        num_ten_elements = np.prod(ten_shape)

        # Values for the two tensors
        ten1 = np.arange(num_ten_elements).reshape(ten_shape)
        ten2 = np.arange(num_ten_elements,
                         2 * num_ten_elements).reshape(ten_shape)

        # Adding the vectors and tensors to omtools
        vec1 = self.declare_input('vec1', val=vec1)
        vec2 = self.declare_input('vec2', val=vec2)

        ten1 = self.declare_input('ten1', val=ten1)
        ten2 = self.declare_input('ten2', val=ten2)

        # Vector-Vector Dot Product
        self.register_output('VecVecDot', ot.dot(vec1, vec2))

        # Tensor-Tensor Dot Product specifying the first axis
        self.register_output('TenTenDotFirst', ot.dot(ten1, ten2, axis=0))

        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('TenTenDotLast', ot.dot(ten1, ten2, axis=2))


prob = Problem()
prob.model = ExampleDot()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('VecVecDot', prob['VecVecDot'].shape)
print(prob['VecVecDot'])
print('TenTenDotFirst', prob['TenTenDotFirst'].shape)
print(prob['TenTenDotFirst'])
print('TenTenDotLast', prob['TenTenDotLast'].shape)
print(prob['TenTenDotLast'])
