from omtools.core.expression import Expression
from openmdao.api import Group
from openmdao.core.system import System


class Subsystem(Expression):
    """
    Class for declaring an input variable
    """
    def initialize(
        self,
        name: str,
        subsys: System,
        promotes=None,
        promotes_inputs=None,
        promotes_outputs=None,
        min_procs=1,
        max_procs=None,
        proc_weight=1.0,
    ):
        self.name: str = name
        self.promotes = promotes
        self.promotes_inputs = promotes_inputs
        self.promotes_outputs = promotes_outputs
        self.num_inputs = 0

        self.build = lambda name: subsys

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        return "Subsystem ('" + self.name + "')"
