from openmdao.api import Problem, IndepVarComp
import numpy as np
import omtools.api as ot
from omtools.api import Group
from omtools.comps.normcomp import NormComp



n = 10
val = np.arange(10)
indeps = IndepVarComp()
indeps.add_output(
    'x',
    val=val,
    shape=(n, ),
)
prob = Problem()
prob.model = Group()
prob.model.add_subsystem(
    'indeps',
    indeps,
    promotes=['*'],
)
prob.model.add_subsystem(
    'two_norm',
    NormComp(in_name='x', out_name='y', shape=(n, ), norm_type=2),
    promotes=['*'],
)
prob.setup()
prob.check_partials(compact_print=True)
prob.run_model()