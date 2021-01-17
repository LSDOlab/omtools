from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import omtools.examples.ex_rotation_matrix as example

# Shape of a random tensor rotation matrix
shape = (2, 3, 4)

num_elements = np.prod(shape)

# Tensor of angles in radians
angle_val1 = np.repeat(np.pi / 3, num_elements).reshape(shape)

angle_val2 = np.repeat(np.pi / 3,
                       num_elements) + 2 * np.pi * np.arange(num_elements)

angle_val2 = angle_val2.reshape(shape)

angle_val3 = np.pi / 3

c = np.cos(np.pi / 3)
s = np.sin(np.pi / 3)

rotmatx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
rotmaty = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

ten_rotmatx = np.tile(rotmatx.flatten(), num_elements).reshape(shape + (3, 3))
print(ten_rotmatx.shape)


def test_rotation_scalar():

    desired_outputx = rotmatx
    np.testing.assert_almost_equal(example.prob['scalar_Rot_x'],
                                   desired_outputx)

    partials_error = example.prob.check_partials(
        includes=['comp_scalar_Rot_x'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)

    desired_outputy = rotmaty
    np.testing.assert_almost_equal(example.prob['scalar_Rot_y'],
                                   desired_outputy)

    partials_error = example.prob.check_partials(
        includes=['comp_scalar_Rot_y'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-6, rtol=1.e-6)


def test_rotation_tensor():

    desired_output = ten_rotmatx
    np.testing.assert_almost_equal(example.prob['tensor1_Rot_x'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_tensor1_Rot_x'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)

    np.testing.assert_almost_equal(example.prob['tensor2_Rot_x'],
                                   desired_output)

    partials_error = example.prob.check_partials(
        includes=['comp_tensor2_Rot_x'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error, atol=1.e-5, rtol=1.e-5)
