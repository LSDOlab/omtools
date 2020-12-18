from openmdao.api import NonlinearBlockGS, ScipyKrylov
from openmdao.api import Problem

import omtools.api as ot
from omtools.api import Group
from omtools.core.expression import Expression


class Example(Group):
    def setup(self):
        x = self.declare_input('x', val=3)
        y = -2 * x**2 + 4 * x + 3
        self.register_output('y', y)


prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.run_model()
