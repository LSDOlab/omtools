from omtools.comps.array_expansion_comp import ArrayExpansionComp
from omtools.comps.scalar_expansion_comp import ScalarExpansionComp
from omtools.core.expression import Expression
from omtools.utils.miscellaneous_functions.decompose_shape_tuple import decompose_shape_tuple


class expand(Expression):
    def initialize(self, expr: Expression, shape: tuple, indices=None):
        if not isinstance(expr, Expression):
            raise TypeError(expr, " is not an Expression object")

        if indices is not None:
            if not isinstance(indices, str):
                raise TypeError(indices, " is not a str or None")

            if '->' not in indices:
                raise ValueError(indices, " is invalid")

        if indices is not None:
            in_indices, out_indices = indices.split('->')

            expand_indices = []
            for i in range(len(out_indices)):
                index = out_indices[i]

                if index not in in_indices:
                    expand_indices.append(i)

        self.shape = shape
        self.add_predecessor_node(expr)

        expr_shape = expr.shape
        if expr_shape == 1:
            expr_shape = (1,)

        if not expr_shape == (1,):
            if indices is None:
                raise ValueError(
                    'If expanding something other than a scalar ' +
                    'indices must be given'
                )

            (
                _, _, _,
                in_shape, _, out_shape,
            ) = decompose_shape_tuple(shape, expand_indices)

            if in_shape != expr_shape:
                raise ValueError(
                    'Shape or indices is invalid'
                )

            self.build = lambda name: ArrayExpansionComp(
                shape=shape,
                expand_indices=expand_indices,
                in_name=expr.name,
                out_name=name,
            )
        else:
            # if indices is not None:
            #     raise ValueError(
            #         'If expanding a scalar ' +
            #         'indices must not be given'
            #     )

            self.build = lambda name: ScalarExpansionComp(
                shape=shape,
                in_name=expr.name,
                out_name=name,
            )