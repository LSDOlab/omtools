from omtools.comps.einsum_comp_dense_derivs import EinsumComp
from omtools.utils.compute_einsum_shape import compute_einsum_shape
from omtools.core.expression import Expression
from typing import List


class einsum(Expression):
    def initialize(self, subscripts: str, *operands: List[Expression]):
        shape = compute_einsum_shape(subscripts)
        self.shape = shape
        for expr in operands:
            if isinstance(expr, Expression) == False:
                raise TypeError(expr, " is not an Expression object")
            self.add_predecessor_node(expr)

        self.build = lambda name: EinsumComp(
            in_names=[expr.name for expr in operands],
            in_shapes=[expr.shape for expr in operands],
            out_name=name,
            operation=subscripts,
            out_shape=shape,
            compute_out_shape=False,
        )
