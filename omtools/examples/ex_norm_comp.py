from openmdao.api import NonlinearBlockGS, ScipyKrylov, NewtonSolver
from openmdao.api import Problem
import numpy as np
import omtools.api as ot
from omtools.api import Group
from omtools.core.expression import Expression


class Example(Group):
    def setup(self):
        n = 10
        m = 20
        V = self.declare_input('V', val=np.arange(n))                     # A vector to test vector norms
        M = self.declare_input('M', val=np.arange(n* m).reshape((n, m)))  # A matrix to test matrix and axis-wise vector norms

        self.register_output('vector_norm_2', ot.norm(V, 2))
        self.register_output('matrix_norm_fro', ot.norm(M, 'fro'))
        self.register_output('axis_0_norm_2', ot.norm(M, 2, axis=0))
        self.register_output('axis_1_norm_2', ot.norm(M, 2, axis=1))
        

prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.check_partials(compact_print=True)
prob.run_model()
        