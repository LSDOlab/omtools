from omtools.comps.einsum_comp_dense_derivs import EinsumComp
from omtools.comps.einsum_comp_sparse_derivs import SparsePartialEinsumComp
from omtools.utils.einsum_utils import compute_einsum_shape, new_einsum_subscripts_to_string_and_list
from omtools.core.expression import Expression
from typing import List


def einsum_new_api(*operands: List[Expression],
                   operation: List[tuple],
                   partial_format='dense'):
    out = Expression()
    for expr in operands:
        if not isinstance(expr, Expression):
            raise TypeError(expr, " is not an Expression object")
        out.add_dependency_node(expr)
    scalar_output = False
    if len(operands) == len(operation):
        scalar_output = True
    operation_aslist, operation_string = new_einsum_subscripts_to_string_and_list(
        operation, scalar_output=scalar_output)

    shape = compute_einsum_shape(operation_aslist,
                                 [expr.shape for expr in operands])
    out.shape = shape

    if partial_format == 'dense':
        out.build = lambda: EinsumComp(
            in_names=[expr.name for expr in operands],
            in_shapes=[expr.shape for expr in operands],
            out_name=out.name,
            operation=operation_string,
            out_shape=shape,
        )
    elif partial_format == 'sparse':
        out.build = lambda: SparsePartialEinsumComp(
            in_names=[expr.name for expr in operands],
            in_shapes=[expr.shape for expr in operands],
            out_name=out.name,
            operation=operation_string,
            out_shape=shape,
        )
    else:
        raise Exception('partial_format should be either dense or sparse')
    return out
