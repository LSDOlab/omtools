from openmdao.api import NonlinearBlockGS, ScipyKrylov, NewtonSolver
from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group
from omtools.core.expression import Expression


class Example(Group):
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
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.run_model()
# print('arccos', prob['arccos'])
# print('arcsin', prob['arcsin'])
# print('arctan', prob['arctan'])
# print('cos', prob['cos'])
# print('cosec', prob['cosec'])
# print('cosech', prob['cosech'])
# print('cosh', prob['cosh'])
# print('cotan', prob['cotan'])
# print('cotanh', prob['cotanh'])
# print('exp', prob['exp'])
# print('log', prob['log'])
# print('log10', prob['log10'])
# print('sec', prob['sec'])
# print('sech', prob['sech'])
# print('sin', prob['sin'])
# print('sinh', prob['sinh'])
# print('tan', prob['tan'])
# print('tanh', prob['tanh'])
