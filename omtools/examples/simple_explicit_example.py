from openmdao.api import NonlinearBlockGS, ScipyKrylov

import omtools.api as ot
from omtools.api import Group
from omtools.core.expression import Expression


class MyGroup(Group):
    def setup(self):
        # Simple addition
        x1 = self.create_indep_var('x1', val=10, dv=True)
        x2 = self.declare_input('x2', val=3)
        self.register_output('y', x1 + x2)


if __name__ == "__main__":
    from openmdao.api import Problem
    import numpy as np

    prob = Problem()
    prob.model = MyGroup()
    prob.setup()
    print('x1 =', prob['x1'])
    print('x2 =', prob['x2'])
    print('y =', prob['y'])
    prob.run_model()
    # prob.check_partials(compact_print=True)
    prob.list_problem_vars()
    print('x1 =', prob['x1'])
    print('x2 =', prob['x2'])
    print('y =', prob['y'])
