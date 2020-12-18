import numpy as np
from openmdao.api import Problem, IndepVarComp
from omtools.comps.einsum_comp_dense_derivs_new_api import EinsumComp as EinsumComp

shape1 = (5, 4)
shape2 = (4,)

a = np.arange(20).reshape((5, 4))
b = np.arange(4)

prob = Problem()

comp = IndepVarComp()
comp.add_output('x', a)
comp.add_output('y', b)
prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

comp = EinsumComp(
    in_names = ['x', 'y'],
    in_shapes = [(5, 4), (4,)],
    out_name = 'f',
    operation = [('row', 'col'), ('col',), ('row',)],
)

prob.model.add_subsystem('comp', comp, promotes=['*'])

prob.setup(check=True)
prob.run_model()
prob.check_partials(compact_print=True)