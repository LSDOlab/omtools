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


        tensor = np.arange(np.prod(shape)).reshape(shape)
        vector = np.arange(np.prod(shape))

        in1 = self.declare_input('in1', val=tensor)
        in2 = self.declare_input('in2', val=vector)

        self.register_output('reshape_tensor2vector', ot.reshape(in1, new_shape=(np.prod(shape),) ) )
        self.register_output('reshape_vector2tensor', ot.reshape(in2, new_shape=shape ) )

        

prob = Problem()
prob.model = Example()
prob.setup(force_alloc_complex=True)
prob.check_partials(compact_print=True)
prob.run_model()
        