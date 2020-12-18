from openmdao.api import NonlinearBlockGS, ScipyKrylov
from openmdao.api import Problem

import omtools.api as ot
from omtools.api import Group
from omtools.core.expression import Expression


class Example(Group):
    def setup(self):
        # Create independent variable
        x1 = self.create_indep_var('x1', val=40)

        # Powers
        y4 = x1**2

        # Create subsystem that depends on previously created
        # independent variable
        group = Group()
        # This value is overwritten by connection
        a = group.declare_input('x1', val=2)
        b = group.create_indep_var('x2', val=12)
        group.register_output('prod', a * b)
        self.add_subsystem('sys', group, promotes=['*'])

        # declare inputs with default values
        # This value is overwritten by connection
        x2 = self.declare_input('x2', val=3)

        # Simple addition
        y1 = x2 + x1
        self.register_output('y1', y1)

        # Simple subtraction
        self.register_output('y2', x2 - x1)

        # Simple multitplication
        self.register_output('y3', x1 * x2)

        # Powers
        y5 = x2**2

        # register outputs in reverse order to how they are defined
        self.register_output('y5', y5)
        self.register_output('y6', y1 + y5)
        self.register_output('y4', y4)


prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.run_model()
