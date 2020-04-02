import kappa_agreement


def test_kappa_equals_1_if_all_same():
    # same file, so maximal agreement
    kappa_dict = kappa_agreement.compute_agreement("annotations/annotator_1.csv",
                                                   "annotations/annotator_1.csv")
    assert(all(value == 1 for value in kappa_dict.values()))
