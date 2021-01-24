from openmdao.api import Problem
from omtools.api import Group
import omtools.api as ot
import numpy as np


class ExampleSingleVector(Group):
    def setup(self):
        n = 3

        # Declare a vector of length 3 as input
        v1 = self.declare_input('v1', val=np.arange(n))

        # Output the average of all the elements of the vector v1
        self.register_output('single_vector_average', ot.average(v1))


prob = Problem()
prob.model = ExampleSingleVector()
prob.setup(force_alloc_complex=True)
prob.run_model()
