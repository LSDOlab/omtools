import numpy as np
from copy import deepcopy

from openmdao.api import ExplicitComponent

from omtools.utils.miscellaneous_functions.process_options import name_types, get_names_list, shape_types, get_shapes_list


class EinsumComp(ExplicitComponent):
    """
    This component computes the Einstein summation convention applied on input components that are arrays.
    Partial derivatives computed are always stored in dense format.

    Options
    -------
    in_names: name_types (str or list of str)
        Input component names that represent input arrays.
        Can be a string or a list with a single string when there is only one input array.
    in_shapes: shape_types (tuple or list of tuples)
        Input component shapes that, in order, represents the shapes of the input arrays.
        Can be a tuple or a list with a single tuple when there is only one input array.
    operation: list of tuples
        Einstein summation operation written as a list of tuples where each tensor is represented by a tuple. Each axis (subscript) of the tensor is named using a string and all the strings are grouped into a tuple. The last tuple should represent the the output array.
    out_name: str
        Output component name that represents the output array calculated based on the Einstein summation operation given.
    """
    def initialize(self):
        self.options.declare('in_names',
                             default=None,
                             types=name_types,
                             allow_none=True)
        self.options.declare('out_name', types=str)
        self.options.declare('in_shapes', types=shape_types)
        # Nametypes might be a string or a list
        # self.options.declare('operation', types=str)
        self.options.declare('operation', types=list)

        self.post_initialize()

    def post_initialize(self):
        pass

    def pre_setup(self):
        pass

    # Add inputs and output, and declare partials
    def setup(self):
        self.pre_setup()
        operation = self.options['operation']

        # Changes from a string to a list with one element if there was only one input
        self.options['in_names'] = get_names_list(self.options['in_names'])
        self.options['in_shapes'] = get_shapes_list(self.options['in_shapes'])

        in_names = self.options['in_names']
        in_shapes = self.options['in_shapes']
        out_name = self.options['out_name']

        # Assign characters to each axis_name in the tuples
        unused_chars = 'abcdefghijklmnopqrstuvwxyz'
        axis_map = {}
        self.operation_as_string = ''
        self.operation_aslist = []
        i = -1
        for axis_names in operation[:-1]:
            tensor_rep = ''
            for axis in axis_names:
                if not (axis in axis_map):
                    axis_map[axis] = unused_chars[0]
                    unused_chars = unused_chars[1:]
                tensor_rep += axis_map[axis]
            self.operation_as_string += tensor_rep
            self.operation_as_string += ','
            self.operation_aslist.append(tensor_rep)

        tensor_rep = ''
        for axis in operation[-1]:
            tensor_rep += axis_map[axis]
        self.operation_as_string = self.operation_as_string[:-1] + '->'
        self.operation_as_string += tensor_rep
        self.operation_aslist.append(tensor_rep)

        self.unused_chars = unused_chars
        self.axis_map = axis_map

        # # Find unused characters in operation
        # check_string = 'abcdefghijklmnopqrstuvwxyz'
        # self.unused_chars = ''
        # for char in check_string:
        #     if not(char in operation):
        #         self.unused_chars += char

        # # Translate the operation string into a list
        # self.operation_aslist = []

        # # Representation of each tensor in the operation string
        # tensor_rep = ''
        # for char in operation:
        #     if char.isalpha():
        #         tensor_rep += char
        #     elif (char == ',' or char == '-'):
        #         self.operation_aslist.append(tensor_rep)
        #         tensor_rep = ''
        # self.operation_aslist.append(tensor_rep)
        '''
        String parse to find output shape
        '''
        out_shape = []
        for char in self.operation_aslist[-1]:
            i = -1
            for tensor_rep in self.operation_aslist[:-1]:
                i += 1
                if (char in tensor_rep):
                    shape_ind = tensor_rep.index(char)
                    out_shape.append(in_shapes[i][shape_ind])
                    break
        self.out_shape = tuple(out_shape)
        self.add_output(out_name, shape=self.out_shape)

        completed_in_names = []
        self.I = []
        operation_aslist = self.operation_aslist

        for in_name_index, in_name in enumerate(in_names):
            if in_name in completed_in_names:
                continue
            else:
                completed_in_names.append(in_name)
            self.add_input(in_name, shape=in_shapes[in_name_index])
            self.declare_partials(out_name, in_name)

            shape = in_shapes[in_name_index]
            size = np.prod(shape)
            rank = len(shape)
            flat_indices = np.arange(size)
            ind = np.unravel_index(flat_indices, shape)
            # self.list_of_tuple_of_indices_of_input_tensors.append(ind)

            # Generate I efficiently for each in_name

            I_shape = 2 * list(shape)
            I_shape = tuple(I_shape)
            I_ind = 2 * list(ind)
            I_ind = tuple(I_ind)

            I = np.zeros(I_shape)
            I[I_ind] += 1

            self.I.append(I)

    def compute(self, inputs, outputs):
        in_names = self.options['in_names']
        out_name = self.options['out_name']
        # operation = self.options['operation']

        outputs[out_name] = np.einsum(
            self.operation_as_string,
            *(inputs[in_name] for in_name in in_names))

    def compute_partials(self, inputs, partials):
        in_names = self.options['in_names']
        in_shapes = self.options['in_shapes']
        out_name = self.options['out_name']
        # operation = self.options['operation']
        operation = self.operation_as_string

        unused_chars = self.unused_chars
        operation_aslist = self.operation_aslist

        completed_in_names = []

        for in_name_index, in_name in enumerate(in_names):
            '''Checking if we are at a repeated input whose derivative was computed at its first occurence in the in_names. If true, we will skip the current iteration of in_name'''
            if in_name in completed_in_names:
                continue
            else:
                completed_in_names.append(in_name)

            shape = in_shapes[in_name_index]
            size = np.prod(shape)
            rank = len(shape)

            # Calculate new_operation for each in_name in first location

            # Compute the locations where the same input is used
            locations = []
            for idx, same_name in enumerate(in_names):
                if same_name == in_name:
                    locations.append(idx)

            new_in_name_tensor_rep = operation_aslist[in_name_index]
            new_in_name_tensor_rep += unused_chars[:rank]
            new_output_tensor_rep = operation_aslist[-1]
            new_output_tensor_rep += unused_chars[:rank]

            new_operation_aslist = deepcopy(operation_aslist)
            new_operation_aslist[in_name_index] = new_in_name_tensor_rep
            new_operation_aslist[-1] = new_output_tensor_rep

            # Compute new_in_names by replacing each in_name in first location by I
            new_operation = ''
            for string_rep in new_operation_aslist[:-1]:
                new_operation += string_rep
                new_operation += ','
            new_operation = new_operation[:-1] + '->'
            new_operation += new_operation_aslist[-1]

            partials[out_name, in_name] = np.einsum(
                new_operation,
                *(inputs[in_name] for in_name in in_names[:in_name_index]),
                self.I[in_name_index],
                *(inputs[in_name] for in_name in in_names[in_name_index + 1:]))

            for i in locations[1:]:
                new_operation_aslist = deepcopy(operation_aslist)
                new_operation_aslist[
                    i] = operation_aslist[i] + unused_chars[:rank]
                new_operation_aslist[-1] = new_output_tensor_rep

                new_operation = ''
                for string_rep in new_operation_aslist[:-1]:
                    new_operation += string_rep
                    new_operation += ','
                new_operation = new_operation[:-1] + '->'
                new_operation += new_operation_aslist[-1]

                partials[out_name, in_name] += np.einsum(
                    new_operation,
                    *(inputs[in_name] for in_name in in_names[:i]),
                    self.I[len(completed_in_names) - 1],
                    *(inputs[in_name]
                      for in_name in in_names[i + 1:])).reshape(
                          partials[out_name, in_name].shape)


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    shape1 = (2, 2, 4)
    shape2 = (2, 7, 4)
    shape3 = (7, 2, 4)

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', np.random.rand(*shape1))
    comp.add_output('y', np.random.rand(*shape2))
    comp.add_output('z', np.random.rand(*shape3))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])
    # out_shape = (2, 3, 4, 4, 7)

    comp = EinsumComp(
        in_names=['x', 'y', 'x'],
        in_shapes=[(2, 2, 4), (2, 7, 4), (2, 2, 4)],
        # in_shapes = [(2, 2, 4), (2, 7, 4), (7, 2, 4)],
        out_name='f',
        # operation = 'abc,ade,fae->abcdf',
        # operation = 'abc,ade,fae->bcdfa',
        # operation = [('row', 'col', 'cat'), ('row','dog','0'), ('46','row','0'),('row', 'col', 'cat','dog','46')],
        operation=[('row', 'col', 'cat'), ('row', 'dog', '0'),
                   ('46', 'row', '0'), ('col', 'cat', 'dog', '46', 'row')],
    )
    # print(np.einsum('abc,ade,fae->abcdf', np.random.rand(*shape1), np.random.rand(*shape2), np.random.rand(*shape3)))
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.check_partials(compact_print=True)
