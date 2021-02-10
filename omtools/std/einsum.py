from omtools.comps.einsum_comp_dense_derivs import EinsumComp
from omtools.comps.einsum_comp_sparse_derivs import SparsePartialEinsumComp
from omtools.utils.einsum_utils import compute_einsum_shape, einsum_subscripts_tolist
from omtools.core.expression import Expression
from typing import List


def einsum(*operands: List[Expression],
           subscripts: str,
           partial_format='dense'):
    out = Expression()
    for expr in operands:
        if not isinstance(expr, Expression):
            raise TypeError(expr, " is not an Expression object")
        out.add_dependency_node(expr)
    operation_aslist = einsum_subscripts_tolist(subscripts)
    shape = compute_einsum_shape(operation_aslist,
                                 [expr.shape for expr in operands])
    out.shape = shape

    if partial_format == 'dense':
        out.build = lambda: EinsumComp(
            in_names=[expr.name for expr in operands],
            in_shapes=[expr.shape for expr in operands],
            out_name=out.name,
            operation=subscripts,
            out_shape=shape,
        )
    elif partial_format == 'sparse':
        out.build = lambda: SparsePartialEinsumComp(
            in_names=[expr.name for expr in operands],
            in_shapes=[expr.shape for expr in operands],
            out_name=out.name,
            operation=subscripts,
            out_shape=shape,
        )
    else:
        raise Exception('partial_format should be either dense or sparse')
    return out
