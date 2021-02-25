from openmdao.api import Problem
from openmdao.api import Problem
from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from omtools.api import Group, ImplicitComponent
import omtools.api as ot
import numpy as np


class ExampleLiterals(Group):
    def setup(self):
        # x = self.create_indep_var('x', shape=(3, 4))
        # y = self.declare_input('y', val=4)
        # z = ot.if_else(
        #     x,
        #     2 * x,
        #     2 * y,
        # )

        # x = self.create_indep_var('x', val=3)
        # y = self.declare_input('y', val=4)
        # z, _ = ot.if_else(
        #     x,
        #     (x, y),
        #     (x, y),
        # )

        x = self.declare_input('x', val=3)
        y = self.declare_input('y', val=4)
        z = ot.if_else(
            2 * x - 8,
            2 * x,
            3 * y,
        )
        self.register_output('z', z)


prob = Problem()
prob.model = ExampleLiterals()
prob.setup(force_alloc_complex=True)
prob.run_model()
