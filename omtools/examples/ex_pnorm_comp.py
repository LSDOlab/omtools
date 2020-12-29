from openmdao.api import NonlinearBlockGS, ScipyKrylov, NewtonSolver
from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group
from omtools.core.expression import Expression


class Example(Group):
    def setup(self):
        i = 2
        j = 3
        k = 4
        l = 5
        shape = (i, j, k, l)
        axis  = (1, 3)

        val = np.arange(np.prod(shape)).reshape(shape)

        in1 = self.declare_input('in1', val=val)
 
        self.register_output('axis_free_pnorm', ot.pnorm(in1, pnorm_type=6))
        self.register_output('axiswise_pnorm', ot.pnorm(in1, axis=axis, pnorm_type=6))

        

prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.check_partials(compact_print=True)
prob.run_model()
        