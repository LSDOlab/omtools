from omtools.comps.reshape_comp import ReshapeComp
from omtools.core.expression import Expression


class reshape(Expression):
    def initialize(self, expr, new_shape):
        if isinstance(expr, Expression):
            self.shape = new_shape
            self.add_predecessor_node(expr)

            self.build = lambda name: ReshapeComp(
                shape=expr.shape,
                in_name=expr.name,
                out_name=name,
                new_shape=self.shape,
            )

        else:
            raise TypeError(expr, " is not an Expression object")
