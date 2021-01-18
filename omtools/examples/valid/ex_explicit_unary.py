from openmdao.api import Problem
from openmdao.api import ScipyKrylov, NewtonSolver, NonlinearBlockGS
from omtools.api import Group, ImplicitComponent
import omtools.api as ot
import numpy as np


class ExampleUnary(Group):
    def setup(self):
        x = self.declare_input('x', val=np.pi)
        y = self.declare_input('y', val=1)
        self.register_output('arccos', ot.arccos(y))
        self.register_output('arcsin', ot.arcsin(y))
        self.register_output('arctan', ot.arctan(x))
        self.register_output('cos', ot.cos(x))
        self.register_output('cosec', ot.cosec(y))
        self.register_output('cosech', ot.cosech(x))
        self.register_output('cosh', ot.cosh(x))
        self.register_output('cotan', ot.cotan(y))
        self.register_output('cotanh', ot.cotanh(x))
        self.register_output('exp', ot.exp(x))
        self.register_output('log', ot.log(x))
        self.register_output('log10', ot.log10(x))
        self.register_output('sec', ot.sec(x))
        self.register_output('sech', ot.sech(x))
        self.register_output('sin', ot.sin(x))
        self.register_output('sinh', ot.sinh(x))
        self.register_output('tan', ot.tan(x))
        self.register_output('tanh', ot.tanh(x))


prob = Problem()
prob.model = ExampleUnary()
prob.setup(force_alloc_complex=True)
prob.run_model()
