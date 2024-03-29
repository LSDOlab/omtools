import numpy as np
from copy import deepcopy
from openmdao.api import ExplicitComponent
from omtools.utils.process_options import name_types, get_names_list, shape_types, get_shapes_list


class SingleTensorAverageComp(ExplicitComponent):
    """
    This component can output multiple variants of average of elements inside a single array:
    Option 1: If the input is a single tensor of any dimension, without any axis provided, the component computes the average of all the entries of the tensor. Output is a scalar.
    Option 2: If the input is a tensor of any dimension, with axis/axes specified, the component computes the average of all the elements of the tensor along the given axis/axes. Output is a tensor whose shape correspond to the shape of the input tensor with the size/s of the specified axis/axes removed.

    Options
    -------
    in_name: str
        Input component name that represent input array.
    shape: tuple
        Shape of the input array.
    axes: tuple
        Axis/axes along which the average is computed for the input array.
    out_name: str
        Output component name that represents the averaged output (can be a scalar or a tensor).
    """
    def initialize(self):
        self.options.declare('in_name',
                             default=None,
                             types=str,
                             allow_none=True)
        self.options.declare('out_name', types=str)
        self.options.declare('shape', types=tuple)
        self.options.declare('axes', default=None, types=tuple)
        self.options.declare('out_shape', default=None, types=tuple)
        self.options.declare('val', types=np.ndarray)

    def setup(self):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']
        out_shape = self.options['out_shape']
        axes = self.options['axes']
        val = self.options['val']

        # Computation of Output shape if the shape is not provided
        if out_shape != None:
            self.output_shape = out_shape
        elif axes != None:
            output_shape = np.delete(shape, axes)
            self.output_shape = tuple(output_shape)

        input_size = np.prod(shape)

        # axes == None works for any tensor
        self.add_input(in_name, shape=shape, val=val)
        if axes == None:
            self.add_output(out_name)
            val = np.full((input_size, ), 1. / input_size)
            self.declare_partials(out_name, in_name, val=val)

        # axes != None works only for matrices
        else:
            self.add_output(out_name, shape=self.output_shape)
            cols = np.arange(input_size)

            rows = np.unravel_index(np.arange(input_size), shape=shape)
            rows = np.delete(np.array(rows), axes, axis=0)
            rows = np.ravel_multi_index(rows, dims=self.output_shape)

            num_entries_averaged = np.prod(np.array(shape)[axes])
            val = np.full((input_size, ), 1. / num_entries_averaged)
            self.declare_partials(out_name,
                                  in_name,
                                  rows=rows,
                                  cols=cols,
                                  val=val)

    def compute(self, inputs, outputs):
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        axes = self.options['axes']

        # axes == None works for any tensor
        if axes == None:
            outputs[out_name] = np.average(inputs[in_name])

        # axes != None works only for matrices
        else:
            outputs[out_name] = np.average(inputs[in_name], axis=axes)


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp, Group
    n = 20
    m = 1
    val1 = np.arange(n * m).reshape(n, m) + 1
    val2 = np.random.rand(n, m)

    indeps = IndepVarComp()
    indeps.add_output(
        'x',
        val=val1,
        shape=(n, m),
    )

    indeps.add_output(
        'y',
        val=val2,
        shape=(n, m),
    )
    prob = Problem()
    prob.model = Group()
    prob.model.add_subsystem(
        'indeps',
        indeps,
        promotes=['*'],
    )
    prob.model.add_subsystem(
        'average',
        SingleTensorAverageComp(in_name='x',
                                out_name='f',
                                shape=(n, m),
                                axes=(0, )),
        promotes=['*'],
    )
    prob.setup()
    prob.check_partials(compact_print=True)
    prob.run_model()
