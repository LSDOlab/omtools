from omtools.core.variable import Variable
from openmdao.core.system import System


class Subsystem(Variable):
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

        self.build = lambda: subsys

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        return "Subsystem ('" + self.name + "')"
