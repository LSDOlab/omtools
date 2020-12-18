from openmdao.api import NonlinearBlockGS, ScipyKrylov

import omtools.api as ot
from omtools.api import Group
from omtools.core.expression import Expression
from openmdao.api import Problem
import numpy as np


class Example(Group):
    def setup(self):
        # These inputs are unused; no components will be constructed
        a = self.declare_input('a', val=10)
        b = self.declare_input('b', val=5)
        c = self.declare_input('c', val=2)


prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.run_model()
