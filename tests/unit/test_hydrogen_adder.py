import numpy as np
import pytest

from memback.structure_repair.hydrogen_adder_gmx import lipid_name_alignment


def test_lipid_name_alignment_reorders_into_target_order():
    aa_names = ["C", "A", "B"]
    pred_names = ["A", "B", "C"]  # itp/target atom order

    align_arr = lipid_name_alignment(aa_names, pred_names)

    reordered = np.array(aa_names)[align_arr]
    np.testing.assert_array_equal(reordered, pred_names)


def test_lipid_name_alignment_identity_when_orders_match():
    names = ["A", "B", "C"]

    align_arr = lipid_name_alignment(names, names)

    np.testing.assert_array_equal(align_arr, [0, 1, 2])


def test_lipid_name_alignment_raises_for_unknown_target_name():
    with pytest.raises(KeyError):
        lipid_name_alignment(["A", "B"], ["A", "Z"])
