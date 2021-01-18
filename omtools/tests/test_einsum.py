from openmdao.utils.assert_utils import assert_check_partials
import numpy as np
import omtools.examples.ex_einsum as example
import omtools.examples.valid.ex_einsum_old_api as example

shape1 = (4, )
shape2 = (5, 4)
shape3 = (2, 4, 3)

vec = np.arange(4)
mat = np.arange(20).reshape(shape2)
tens = np.arange(24).reshape(shape3)


def test_einsum_inner():
    desired_output1 = np.einsum('i,i->', vec, vec)
    desired_output2 = np.einsum('ijk,j->ik', tens, vec)

    np.testing.assert_array_almost_equal(example.prob['einsum_inner1'],
                                         desired_output1)
    np.testing.assert_array_almost_equal(example.prob['einsum_inner2'],
                                         desired_output2)

    partials_error1 = example.prob.check_partials(
        includes=['comp_einsum_inner1'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error1, atol=1.e-5, rtol=1.e-5)

    partials_error2 = example.prob.check_partials(
        includes=['comp_einsum_inner2'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error2, atol=1.e-5, rtol=1.e-5)


def test_einsum_outer():
    desired_output1 = np.einsum('i,j->ij', vec, vec)
    desired_output2 = np.einsum('hij,k->hijk', tens, vec)

    np.testing.assert_array_almost_equal(example.prob['einsum_outer1'],
                                         desired_output1)
    np.testing.assert_array_almost_equal(example.prob['einsum_outer2'],
                                         desired_output2)

    partials_error1 = example.prob.check_partials(
        includes=['comp_einsum_outer1'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error1, atol=1.e-5, rtol=1.e-5)

    partials_error2 = example.prob.check_partials(
        includes=['comp_einsum_outer2'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error2, atol=1.e-5, rtol=1.e-5)


def test_einsum_reorder():
    desired_output1 = np.einsum('ij->ji', mat)
    desired_output2 = np.einsum('ijk->kji', tens)

    np.testing.assert_array_almost_equal(example.prob['einsum_reorder1'],
                                         desired_output1)
    np.testing.assert_array_almost_equal(example.prob['einsum_reorder2'],
                                         desired_output2)

    partials_error1 = example.prob.check_partials(
        includes=['comp_einsum_reorder1'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error1, atol=1.e-6, rtol=1.e-6)

    partials_error2 = example.prob.check_partials(
        includes=['comp_einsum_reorder2'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error2, atol=1.e-6, rtol=1.e-6)


def test_einsum_summation():
    desired_output1 = np.einsum('i->', vec)
    desired_output2 = np.einsum('ijk->', tens)

    np.testing.assert_array_almost_equal(example.prob['einsum_summ1'],
                                         desired_output1)
    np.testing.assert_array_almost_equal(example.prob['einsum_summ2'],
                                         desired_output2)

    partials_error1 = example.prob.check_partials(
        includes=['comp_einsum_summ1'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error1, atol=1.e-6, rtol=1.e-6)

    partials_error2 = example.prob.check_partials(
        includes=['comp_einsum_summ2'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error2, atol=1.e-6, rtol=1.e-6)


def test_einsum_special():
    desired_output1 = np.einsum('i,j->j', vec, vec)
    desired_output2 = np.einsum('i,j->', vec, vec)

    np.testing.assert_array_almost_equal(example.prob['einsum_special1'],
                                         desired_output1)
    np.testing.assert_array_almost_equal(example.prob['einsum_special2'],
                                         desired_output2)

    partials_error1 = example.prob.check_partials(
        includes=['comp_einsum_special1'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error1, atol=1.e-5, rtol=1.e-5)

    partials_error2 = example.prob.check_partials(
        includes=['comp_einsum_special2'], out_stream=None, compact_print=True)
    assert_check_partials(partials_error2, atol=1.e-5, rtol=1.e-5)


def test_einsum_sparse_inner():
    desired_output1 = np.einsum('i,i->', vec, vec)
    desired_output2 = np.einsum('ijk,j->ik', tens, vec)

    np.testing.assert_array_almost_equal(
        example.prob['einsum_inner1_sparse_derivs'], desired_output1)
    np.testing.assert_array_almost_equal(
        example.prob['einsum_inner2_sparse_derivs'], desired_output2)

    partials_error1 = example.prob.check_partials(
        includes=['comp_einsum_inner1_sparse_derivs'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error1, atol=1.e-5, rtol=1.e-5)

    partials_error2 = example.prob.check_partials(
        includes=['comp_einsum_inner2_sparse_derivs'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error2, atol=1.e-5, rtol=1.e-5)


def test_einsum_sparse_outer():
    desired_output1 = np.einsum('i,j->ij', vec, vec)
    desired_output2 = np.einsum('hij,k->hijk', tens, vec)

    np.testing.assert_array_almost_equal(
        example.prob['einsum_outer1_sparse_derivs'], desired_output1)
    np.testing.assert_array_almost_equal(
        example.prob['einsum_outer2_sparse_derivs'], desired_output2)

    partials_error1 = example.prob.check_partials(
        includes=['comp_einsum_outer1_sparse_derivs'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error1, atol=1.e-5, rtol=1.e-5)

    partials_error2 = example.prob.check_partials(
        includes=['comp_einsum_outer2_sparse_derivs'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error2, atol=1.e-5, rtol=1.e-5)


def test_einsum_sparse_reorder():
    desired_output1 = np.einsum('ij->ji', mat)
    desired_output2 = np.einsum('ijk->kji', tens)

    np.testing.assert_array_almost_equal(
        example.prob['einsum_reorder1_sparse_derivs'], desired_output1)
    np.testing.assert_array_almost_equal(
        example.prob['einsum_reorder2_sparse_derivs'], desired_output2)

    partials_error1 = example.prob.check_partials(
        includes=['comp_einsum_reorder1_sparse_derivs'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error1, atol=1.e-6, rtol=1.e-6)

    partials_error2 = example.prob.check_partials(
        includes=['comp_einsum_reorder2_sparse_derivs'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error2, atol=1.e-6, rtol=1.e-6)


def test_einsum_sparse_summation():
    desired_output1 = np.einsum('i->', vec)
    desired_output2 = np.einsum('ijk->', tens)

    np.testing.assert_array_almost_equal(
        example.prob['einsum_summ1_sparse_derivs'], desired_output1)
    np.testing.assert_array_almost_equal(
        example.prob['einsum_summ2_sparse_derivs'], desired_output2)

    partials_error1 = example.prob.check_partials(
        includes=['comp_einsum_summ1_sparse_derivs'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error1, atol=1.e-6, rtol=1.e-6)

    partials_error2 = example.prob.check_partials(
        includes=['comp_einsum_summ2_sparse_derivs'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error2, atol=1.e-6, rtol=1.e-6)


def test_einsum_sparse_special():
    desired_output1 = np.einsum('i,j->j', vec, vec)
    desired_output2 = np.einsum('i,j->', vec, vec)

    np.testing.assert_array_almost_equal(
        example.prob['einsum_special1_sparse_derivs'], desired_output1)
    np.testing.assert_array_almost_equal(
        example.prob['einsum_special2_sparse_derivs'], desired_output2)

    partials_error1 = example.prob.check_partials(
        includes=['comp_einsum_special1_sparse_derivs'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error1, atol=1.e-5, rtol=1.e-5)

    partials_error2 = example.prob.check_partials(
        includes=['comp_einsum_special2_sparse_derivs'],
        out_stream=None,
        compact_print=True)
    assert_check_partials(partials_error2, atol=1.e-5, rtol=1.e-5)
