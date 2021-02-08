from omtools.comps.einsum_comp_dense_derivs import EinsumComp
from omtools.comps.einsum_comp_sparse_derivs import SparsePartialEinsumComp
from omtools.utils.einsum_utils import compute_einsum_shape, einsum_subscripts_tolist
from omtools.core.expression import Expression
from typing import List


class einsum(Expression):
    def initialize(self,
                   *operands: List[Expression],
                   subscripts: str,
                   partial_format='dense'):

        for expr in operands:
            if isinstance(expr, Expression) == False:
                raise TypeError(expr, " is not an Expression object")
            self.add_predecessor_node(expr)

        operation_aslist = einsum_subscripts_tolist(subscripts)
        shape = compute_einsum_shape(operation_aslist,
                                     [expr.shape for expr in operands])
        self.shape = shape

        if partial_format == 'dense':
            self.build = lambda: EinsumComp(
                in_names=[expr.name for expr in operands],
                in_shapes=[expr.shape for expr in operands],
                out_name=self.name,
                operation=subscripts,
                out_shape=shape,
            )
        elif partial_format == 'sparse':
            self.build = lambda: SparsePartialEinsumComp(
                in_names=[expr.name for expr in operands],
                in_shapes=[expr.shape for expr in operands],
                out_name=self.name,
                operation=subscripts,
                out_shape=shape,
            )
        else:
            raise Exception('partial_format should be either dense or sparse')
