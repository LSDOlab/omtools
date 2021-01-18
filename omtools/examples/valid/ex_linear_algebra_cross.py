from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleCross(Group):
    def setup(self):
        # Creating two vectors
        vecval1 = np.arange(3)
        vecval2 = np.arange(3) + 1

        vec1 = self.declare_input('vec1', val=vecval1)
        vec2 = self.declare_input('vec2', val=vecval2)

        # Vector-Vector Cross Product
        self.register_output('VecVecCross', ot.cross(vec1, vec2, axis=0))

        # Creating two tensors
        shape = (2, 5, 4, 3)
        num_elements = np.prod(shape)

        tenval1 = np.arange(num_elements).reshape(shape)
        tenval2 = np.arange(num_elements).reshape(shape) + 6

        ten1 = self.declare_input('ten1', val=tenval1)
        ten2 = self.declare_input('ten2', val=tenval2)

        # Tensor-Tensor Dot Product specifying the last axis
        self.register_output('TenTenCross', ot.cross(ten1, ten2, axis=3))


prob = Problem()
prob.model = ExampleCross()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('VecVecCross', prob['VecVecCross'].shape)
print(prob['VecVecCross'])
print('TenTenCross', prob['TenTenCross'].shape)
print(prob['TenTenCross'])
