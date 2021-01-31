from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleMultipleTensor(Group):
    def setup(self):
        n = 3
        m = 6
        p = 7
        q = 10

        # Declare a tensor of shape 3x6x7x10 as input
        T1 = self.declare_input('T1',
                                val=np.arange(n * m * p * q).reshape(
                                    (n, m, p, q)))

        # Declare another tensor of shape 3x6x7x10 as input
        T2 = self.declare_input('T2',
                                val=np.arange(n * m * p * q,
                                              2 * n * m * p * q).reshape(
                                                  (n, m, p, q)))
        # Output the elementwise average of tensors T1 and T2
        self.register_output('multiple_tensor_average', ot.average(T1, T2))


prob = Problem()
prob.model = ExampleMultipleTensor()
prob.setup(force_alloc_complex=True)
prob.run_model()

print('T1', prob['T1'].shape)
print(prob['T1'])
print('T2', prob['T2'].shape)
print(prob['T2'])
print('multiple_tensor_average', prob['multiple_tensor_average'].shape)
print(prob['multiple_tensor_average'])
