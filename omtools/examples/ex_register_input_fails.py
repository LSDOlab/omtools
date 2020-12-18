from openmdao.api import NonlinearBlockGS, ScipyKrylov

import omtools.api as ot
from omtools.api import Group
from omtools.core.expression import Expression
from openmdao.api import Problem
import numpy as np


class Example(Group):
    def setup(self):
        a = self.declare_input('a', val=10)
        # This will raise a TypeError
        self.register_output('a', a)


prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
