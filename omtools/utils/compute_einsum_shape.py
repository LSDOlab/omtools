from typing import List, Union, Tuple


def compute_einsum_shape(
    operation_aslist: List[str],
    in_shapes: Union[Tuple[int], List[Tuple[int]]],
):
    out_shape = []
    for char in operation_aslist[-1]:
        i = -1
        for tensor_rep in operation_aslist[:-1]:
            i += 1
            if (char in tensor_rep):
                shape_ind = tensor_rep.index(char)
                out_shape.append(in_shapes[i][shape_ind])
                break
    return tuple(out_shape)
