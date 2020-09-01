from omtools.components.coscomp import CosComp
from omtools.expression import UnaryExpression


# TODO: Don't define names in two places
class Cos():
    def __init__(self, expr):
        self.name = "cos_BO_" + expr.name + "_BC"
        self.shape = expr.shape

        def build(name):
            return CosComp(
                shape=expr.shape,
                out_name=name,
                in_name=expr.name,
            )

        self.build = build


class cos(UnaryExpression):
    def initialize(self, expr):
        b = Cos(expr)
        self.name = b.name
        self.input_names = [expr.name]
        self.input_shapes = [expr.shape]
        self.shape = expr.shape
        self.builders[self.name] = b
