import numpy as np
from openmdao.api import Problem, IndepVarComp
from omtools.comps.einsum_comp_dense_derivs_new_api import EinsumComp as EinsumComp


shape1 = (5,)
shape2 = (4,)

a = np.arange(5,)
b = np.arange(4,)

prob = Problem()

comp = IndepVarComp()
comp.add_output('x', a)
comp.add_output('y', b)
prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

comp = EinsumComp(
    in_names = ['x', 'y'],
    in_shapes = [(5,), (4,)],
    out_name = 'f',
    operation = [('row',), ('col',), ('row', 'col')],
)

prob.model.add_subsystem('comp', comp, promotes=['*'])

prob.setup(check=True)
prob.run_model()
prob.check_partials(compact_print=True)