import numpy as np
from openmdao.api import ExplicitComponent
from omtools.core.expression import Expression


class IndexedPassThroughComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('expr_indices', types=dict)
        self.options.declare('out_expr', types=Expression)

    def setup(self):
        expr_indices = self.options['expr_indices']
        out_expr = self.options['out_expr']
        self.add_output(
            out_expr.name,
            val=out_expr.val,
            shape=out_expr.shape,
            # units=out_expr.units,
        )

        for expr, flat_dest_indices in expr_indices.items():
            self.add_input(
                expr.name,
                val=expr.val,
                shape=expr.shape,
                # units=expr.units,
            )
            self.declare_partials(
                out_expr.name,
                expr.name,
                val=1.,
                rows=flat_dest_indices,
                cols=np.arange(len(flat_dest_indices)),
            )

    def compute(self, inputs, outputs):
        out_expr = self.options['out_expr']
        expr_indices = self.options['expr_indices']
        for expr, flat_dest_indices in expr_indices.items():
            outputs[out_expr.name][np.unravel_index(
                flat_dest_indices,
                out_expr.shape)] = inputs[expr.name].flatten()
