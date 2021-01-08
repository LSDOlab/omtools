import typing
from typing import Dict, List, Set, Callable

import numpy as np
from openmdao.api import Problem
from openmdao.api import ImplicitComponent as OMImplicitComponent
from openmdao.solvers.solver import Solver

from omtools.core.subsystem import Subsystem
from omtools.core.expression import Expression
from omtools.core.group import Group
from omtools.core.input import Input
from omtools.utils.collect_input_exprs import collect_input_exprs


# TODO: get initial value for ImplicitOutput
def _post_setup(func: Callable) -> Callable:
    def _build_problem(self):
        func(self)
        # setup internal problem
        g = self.group
        for res_expr in g._root.predecessors:
            if isinstance(res_expr, Subsystem) == False and isinstance(
                    res_expr, Input) == False:
                # inputs for this residual only
                in_exprs = set(collect_input_exprs([], res_expr, res_expr))
                # output corresponding to this residual
                out_name = self.group.res_out_map[res_expr.name]

                # accumulate inputs and outputs to set values
                self.all_inputs[out_name] = in_exprs
                self.all_outputs = self.all_outputs.union({out_name})

                # add component output
                # output is an input to internal problem
                print('out', out_name)
                self.add_output(
                    out_name,
                    shape=res_expr.shape,
                    val=res_expr.val,
                )

                # add inputs, declare partials (out wrt in)
                for in_expr in in_exprs:
                    if in_expr.name != out_name:
                        print('in', in_expr.name)
                        self.add_input(
                            in_expr.name,
                            shape=in_expr.shape,
                            val=in_expr.val,
                        )
                        self.declare_partials(
                            of=out_name,
                            wrt=in_expr.name,
                        )

                # residual wrt output
                self.declare_partials(
                    of=out_name,
                    wrt=out_name,
                )

                # TODO: necessary?
                # set response and design variables to compute derivatives
                # treat inputs and output as inputs to internal model
                # treat residual as output of internal model
                self.prob.model.add_constraint(res_expr.name)
                for in_expr in in_exprs:
                    in_name = in_expr.name
                    if in_name not in self.prob.model._design_vars and in_name not in self.prob.model._static_design_vars:
                        self.prob.model.add_design_var(in_name)
                if out_name in self.prob.model._design_vars or out_name in self.prob.model._static_design_vars:
                    pass
                else:
                    self.prob.model.add_design_var(out_name)
        self.prob.setup()

        # set initial values for inputs and output
        for res_expr in self.group._root.predecessors:
            if isinstance(res_expr, Subsystem) == False and isinstance(
                    res_expr, Input) == False:
                out_name = self.group.res_out_map[res_expr.name]
                # TODO: where is initial value for ImplicitOutput?
                self.prob[out_name] = res_expr.val
                for in_expr in self.all_inputs[out_name]:
                    self.prob[in_expr.name] = in_expr.val

        # create n2 diagram of internal model for debugging
        # if self.options['n2'] == True:
        # self.prob.run_model()
        # self.prob.model.list_inputs()
        # self.prob.model.list_outputs()
        # from openmdao.api import n2
        # n2(self.prob)

    return _build_problem


class _ProblemBuilder(type):
    def __new__(cls, name, bases, attr):
        print('replacing setup for implicit comp')
        attr['setup'] = _post_setup(attr['setup'])
        return super(_ProblemBuilder, cls).__new__(cls, name, bases, attr)


class ImplicitComponent(OMImplicitComponent, metaclass=_ProblemBuilder):
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
    def __init__(self, x1=0, x2=1, **kwargs):
        super().__init__(**kwargs)
        self._inst_functs = {
            name: getattr(self, name, None)
            for name in
            ['apply_linear', 'apply_multi_linear', 'solve_multi_linear']
        }

        self.prob = Problem()
        self.prob.model = Group()
        self.group = self.prob.model
        # TODO: modify ImplicitOutput to provide out_expr and brackets
        self.all_inputs: Dict[Set[Input]] = dict()
        self.all_outputs: Set[str] = set()
        self.derivs = dict()
        self.x1 = x1
        self.x2 = x2

    def initialize(self):
        self.options.declare('maxiter', types=int, default=100)
        # self.options.declare('x1')
        # self.options.declare('x2')
        self.options.declare('n2', types=bool, default=False)

    def _set_values(self, inputs, outputs):
        for res_expr in self.group._root.predecessors:
            if isinstance(res_expr, Subsystem) == False:
                out_name = self.group.res_out_map[res_expr.name]
                self.prob[out_name] = outputs[out_name]
                for in_expr in self.all_inputs[out_name]:
                    if in_expr.name != out_name:
                        self.prob[in_expr.name] = inputs[in_expr.name]

    def setup(self):
        pass

    def create_group(self, name: str) -> Group:
        group = Group()
        self.group.add_subsystem(name, group, promotes=['*'])
        return group

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
        self._set_values(inputs, outputs)
        prob = self.prob
        prob.run_model()

        for res_name, out_name in self.group.res_out_map.items():
            residuals[out_name] = np.array(prob[res_name])

    # def solve_nonlinear(self, inputs, outputs):
    #     # TODO: get brackets from ImplicitOutput exprs
    #     for res_expr in self.group._root.predecessors:
    #         out_name = self.group.res_out_map[res_expr.name]
    #         shape = res_expr.shape

    #         x1 = self.x1 * np.ones(shape)
    #         x2 = self.x2 * np.ones(shape)

    #         r1 = self.run(inputs, x1)
    #         r2 = self.run(inputs, x2)
    #         mask1 = r1 >= r2
    #         mask2 = r1 < r2

    #         xp = np.empty(shape)
    #         xp[mask1] = x1[mask1]
    #         xp[mask2] = x2[mask2]

    #         xn = np.empty(shape)
    #         xn[mask1] = x2[mask1]
    #         xn[mask2] = x1[mask2]

    #         for _ in range(self.options['maxiter']):
    #             x = 0.5 * xp + 0.5 * xn
    #             r = self.run(inputs, x)
    #             mask_p = r >= 0
    #             mask_n = r < 0
    #             xp[mask_p] = x[mask_p]
    #             xn[mask_n] = x[mask_n]

    #         outputs[out_name] = 0.5 * xp + 0.5 * xn

    def linearize(self, inputs, outputs, jacobian):
        self._set_values(inputs, outputs)
        in_exprs = set()
        for inputs in self.all_inputs.values():
            in_exprs = in_exprs.union(inputs)
        res_names = list(self.group.res_out_map.keys())
        out_names = list(self.all_inputs.keys())

        jac = self.prob.compute_totals(
            of=res_names,
            wrt=[in_expr.name for in_expr in list(in_exprs)] + out_names,
        )

        for res_expr in self.group._root.predecessors:
            if isinstance(res_expr, Subsystem) == False:
                res_name = res_expr.name
                out_name = self.group.res_out_map[res_name]
                for in_expr in self.all_inputs[out_name]:
                    jacobian[out_name, in_expr.name] = jac[res_name,
                                                           in_expr.name]
                jacobian[out_name, out_name] = jac[res_name, out_name]

                self.derivs[out_name] = np.diag(
                    jac[res_name, out_name]).reshape(res_expr.shape)

    def solve_linear(self, d_outputs, d_residuals, mode):
        out_expr = self.options['out_expr']
        out_name = out_expr.name

        for out_name in self.group.res_out_map.values():
            if mode == 'fwd':
                d_outputs[out_name] += 1. / self.derivs[
                    out_name] * d_residuals[out_name]
            else:
                d_residuals[out_name] += 1. / self.derivs[
                    out_name] * d_outputs[out_name]
