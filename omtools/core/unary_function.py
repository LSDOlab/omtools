from omtools.core.expression import Expression


class UnaryFunction(Expression):
    """
    Base class for classes that represent a function/expression with a
    single argument (e.g. ``exp``, ``sin``, ``cos``)
    """
    def initialize(self, expr: Expression):
        self.predecessors.append(expr)

    def __repr__(self):
        shape_str = "("
        for dim in self.shape:
            shape_str += str(dim) + ","
        shape_str += ")"
        return "UnaryFunction (" + shape_str + ")"
