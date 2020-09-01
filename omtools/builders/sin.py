from omtools.components.sincomp import SinComp
from omtools.expression import UnaryExpression


# TODO: Don't define names in two places
class Sin():
    def __init__(self, expr):
        self.name = "sin_BO_" + expr.name + "_BC"
        self.shape = expr.shape

        def build(name):
            return SinComp(
                shape=expr.shape,
                out_name=name,
                in_name=expr.name,
            )

        self.build = build


class sin(UnaryExpression):
    def initialize(self, expr):
        b = Sin(expr)
        self.name = b.name
        self.input_names = [expr.name]
        self.input_shapes = [expr.shape]
        self.shape = expr.shape
        self.builders[self.name] = b
