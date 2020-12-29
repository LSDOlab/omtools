from omtools.comps.array_expansion_comp import ArrayExpansionComp
from omtools.comps.scalar_expansion_comp import ScalarExpansionComp
from omtools.core.expression import Expression
from omtools.utils.miscellaneous_functions.decompose_shape_tuple import decompose_shape_tuple


class expand(Expression):
    def initialize(self, expr: Expression, shape: tuple, expand_indices=None):
        if not isinstance(expr, Expression):
            raise TypeError(expr, " is not an Expression object")

        if expand_indices is not None:
            if not isinstance(expand_indices, list):
                raise TypeError(expand_indices, " is not a list or None")

        self.shape = shape
        self.add_predecessor_node(expr)

        if not expr.shape == (1,):
            if expand_indices is None:
                raise ValueError(
                    'If expanding something other than a scalar ' +
                    'expand_indices must be given'
                )

            (
                _, _, _,
                in_shape, _, out_shape,
            ) = decompose_shape_tuple(shape, expand_indices)

            if in_shape != expr.shape:
                raise ValueError(
                    'Shape or expand_indices is invalid'
                )

            self.build = lambda name: ArrayExpansionComp(
                shape=shape,
                expand_indices=expand_indices,
                in_name=expr.name,
                out_name=name,
            )
        else:
            if expand_indices is not None:
                raise ValueError(
                    'If expanding a scalar ' +
                    'expand_indices must not be given'
                )

            self.build = lambda name: ScalarExpansionComp(
                shape=shape,
                in_name=expr.name,
                out_name=name,
            )