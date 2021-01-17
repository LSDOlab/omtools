from omtools.comps.cross_product_comp import CrossProductComp
from omtools.core.expression import Expression


class cross(Expression):
    def initialize(self, in1, in2, axis:int):
        if isinstance(in1, Expression) and isinstance(in2, Expression):

            self.add_predecessor_node(in1)
            self.add_predecessor_node(in2)

            if in1.shape != in2.shape:
                raise Exception("The shapes of the inputs must match!")
            else:
                self.shape = in1.shape

            if in1.shape[axis] != 3:
                raise Exception("The specified axis must correspond to the value of 3 in shape")


            self.build = lambda name: CrossProductComp(
                shape=in1.shape,
                in1_name=in1.name,
                in2_name=in2.name,
                out_name=name,            
                axis=axis,
                )

        else:
            raise TypeError(expr, " is not an Expression object")
