import typing
from typing import List

from openmdao.api import ImplicitComponent, Problem
from openmdao.solvers.solver import Solver

from omtools.core.expression import Expression
from omtools.core.input import Input
from omtools.utils.collect_input_exprs import collect_input_exprs


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

        # create Problem and Group containing any number of
        # Components to compute composite residual and partials
        self.prob = Problem()
        self.prob.model = Group()

    def setup(self):
        in_exprs = self.options['in_exprs']
        out_expr = self.options['out_expr']
        res_expr = self.options['res_expr']

        for in_expr in in_exprs:
            self.add_input(in_expr.name, val=in_expr.val)
        self.add_output(out_expr.name, val=out_expr.val)
        self.declare_partials(of='*', wrt='*')

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
            self.prob.set_val(
                in_expr.name,
                in_expr.val,
            )
        self.prob.set_val(
            out_expr.name,
            out_expr.val,
        )

        # self.prob.run_model()
        # self.prob.model.list_inputs()
        # self.prob.model.list_outputs()
        # from openmdao.api import n2
        # n2(self.prob)

    def apply_nonlinear(self, inputs, outputs, residuals):
        in_exprs = self.options['in_exprs']
        out_expr = self.options['out_expr']

        # update input and output values before each iteration
        for in_expr in in_exprs:
            self.prob.set_val(
                in_expr.name,
                inputs[in_expr.name],
            )
        self.prob.set_val(
            out_expr.name,
            outputs[out_expr.name],
        )

        # compute residual
        self.prob.run_model()
        # print(out_expr.name, self.prob[out_expr.name])
        residuals[out_expr.name] = self.prob[out_expr.name]

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
        jac = self.prob.compute_totals()
        for in_expr in in_exprs:
            of_res = 'comp_' + res_expr.name + '.' + res_expr.name
            of_out = out_expr.name
            jacobian[of_out, in_expr.name] = jac[of_res, in_expr.name]
        jacobian[of_out, out_expr.name] = jac[of_res, out_expr.name]
