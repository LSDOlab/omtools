from omtools.api import Group
from openmdao.api import Problem


class Example(Group):
    def setup(self):
        z = self.create_indep_var('z', val=10)


prob = Problem()
prob.model = Example()
prob.setup()
prob.run_model()
