import typing
from typing import List

from openmdao.api import ImplicitComponent, Problem
from openmdao.solvers.solver import Solver

from omtools.core.expression import Expression
from omtools.core.input import Input
from omtools.utils.collect_input_exprs import collect_input_exprs

import numpy as np


class CompositeImplicitComp(ImplicitComponent):
    """
    Class for creating ``ImplicitComponent`` objects that compute
    composite residuals

    Options
    -------
    out_expr: ImplicitOutput
        Object that represents the output of the
        ``CompositeImplicitComp``

    res_expr: Expression
        Object that represents an expression to compute the residual

    """
    def initialize(self):
        # this avoids circular imports
        from omtools.core.implicit_output import ImplicitOutput
        from omtools.core.group import Group
        self.options.declare('in_exprs', types=list)
        self.options.declare('out_expr', types=ImplicitOutput)
        self.options.declare('res_expr', types=Expression)
        self.options.declare('maxiter', types=int, default=100)
        self.options.declare('x1')
        self.options.declare('x2')

        # create Problem and Group containing any number of
        # Components to compute composite residual and partials
        self.prob = Problem()
        self.prob.model = Group()

    def setup(self):
        in_exprs = self.options['in_exprs']
        out_expr = self.options['out_expr']
        res_expr = self.options['res_expr']

        if out_expr.shape != res_expr.shape:
            raise ValueError(
                "Output and residual Expressions must be the same shape")

        # rows = np.arange(np.prod(out_expr.shape))
        for in_expr in in_exprs:
            self.add_input(
                in_expr.name,
                shape=in_expr.shape,
                val=in_expr.val,
            )
            self.declare_partials(
                of=out_expr.name,
                wrt=in_expr.name,
                # rows=rows,
                # cols=np.arange(np.prod(in_expr.shape)),
            )
        self.add_output(
            out_expr.name,
            shape=out_expr.shape,
            val=out_expr.val,
        )
        # residual corresponding to output wrt output
        self.declare_partials(
            of=out_expr.name,
            wrt=out_expr.name,
            # rows=rows,
            # cols=rows,
        )

        # register expression that computes residual
        self.prob.model.register_output(
            res_expr.name,
            res_expr,
        )

        # set response and design variables to compute derivatives
        self.prob.model.add_objective(res_expr.name)
        for in_expr in in_exprs:
            self.prob.model.add_design_var(in_expr.name)
        self.prob.model.add_design_var(out_expr.name)

        # setup problem, constructing components that compute residual
        self.prob.setup()

        # set initial values for inputs and output
        for in_expr in in_exprs:
            self.prob[in_expr.name] = in_expr.val
        self.prob[out_expr.name] = out_expr.val

        # self.prob.run_model()
        # self.prob.model.list_inputs()
        # self.prob.model.list_outputs()
        # from openmdao.api import n2
        # n2(self.prob)

    def run(self, inputs, output):
        in_exprs = self.options['in_exprs']
        out_expr = self.options['out_expr']
        res_expr = self.options['res_expr']
        prob = self.prob

        for in_expr in in_exprs:
            prob[in_expr.name] = inputs[in_expr.name]
        prob[out_expr.name] = output

        prob.run_model()

        return np.array(prob[res_expr.name])

    def apply_nonlinear(self, inputs, outputs, residuals):
        out_expr = self.options['out_expr']
        residuals[out_expr.name] = self.run(
            inputs,
            outputs[out_expr.name],
        )

    def solve_nonlinear(self, inputs, outputs):
        out_expr = self.options['out_expr']
        shape = out_expr.shape

        x1 = self.options['x1'] * np.ones(shape)
        x2 = self.options['x2'] * np.ones(shape)

        r1 = self.run(inputs, x1)
        r2 = self.run(inputs, x2)
        mask1 = r1 >= r2
        mask2 = r1 < r2

        xp = np.empty(shape)
        xp[mask1] = x1[mask1]
        xp[mask2] = x2[mask2]

        xn = np.empty(shape)
        xn[mask1] = x2[mask1]
        xn[mask2] = x1[mask2]

        for _ in range(self.options['maxiter']):
            x = 0.5 * xp + 0.5 * xn
            r = self.run(inputs, x)
            mask_p = r >= 0
            mask_n = r < 0
            xp[mask_p] = x[mask_p]
            xn[mask_n] = x[mask_n]

        outputs[out_expr.name] = 0.5 * xp + 0.5 * xn

    def linearize(self, inputs, outputs, jacobian):
        in_exprs = self.options['in_exprs']
        out_expr = self.options['out_expr']
        res_expr = self.options['res_expr']

        # set input and output values before each iteration
        for in_expr in in_exprs:
            self.prob.set_val(
                in_expr.name,
                inputs[in_expr.name],
            )
        self.prob.set_val(
            out_expr.name,
            outputs[out_expr.name],
        )

        # compute partials
        res_name = res_expr.name
        out_name = out_expr.name

        jac = self.prob.compute_totals(
            of=[res_name],
            wrt=[in_expr.name for in_expr in in_exprs] + [out_expr.name],
        )
        for in_expr in in_exprs:
            jacobian[out_name, in_expr.name] = jac[res_name, in_expr.name]
        jacobian[out_name, out_expr.name] = jac[res_name, out_expr.name]

        self.derivs = np.diag(jac[res_name,
                                  out_expr.name]).reshape(out_expr.shape)

    def solve_linear(self, d_outputs, d_residuals, mode):
        out_expr = self.options['out_expr']
        out_name = out_expr.name

        if mode == 'fwd':
            d_outputs[out_name] += 1. / self.derivs * d_residuals[out_name]
        else:
            d_residuals[out_name] += 1. / self.derivs * d_outputs[out_name]
