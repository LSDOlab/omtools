from openmdao.api import Problem, IndepVarComp
import numpy as np
import omtools.api as ot
from omtools.api import Group
from omtools.comps.normcomp import NormComp



n = 10
m = 20
val = np.arange(n* m).reshape((n, m))
indeps = IndepVarComp()
indeps.add_output(
    'x',
    val=val,
    shape=(n, m),
)
prob = Problem()
prob.model = Group()
prob.model.add_subsystem(
    'indeps',
    indeps,
    promotes=['*'],
)
prob.model.add_subsystem(
    'fro_norm',
    NormComp(in_name='x', out_name='y', shape=(n, m), norm_type='fro'),
    promotes=['*'],
)
prob.setup()
prob.check_partials(compact_print=True)
prob.run_model()