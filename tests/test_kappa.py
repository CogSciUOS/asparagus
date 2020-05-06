import measure_agreement

def test_kappa_equals_1_if_all_same():
    # same file, so maximal agreement
    annotations1, _annotations = measure_agreement.load_annotations("annotations/template_structure_annotation/annotator_1.csv", "annotations/template_structure_annotation/annotator_1.csv")
    kappa_dict = measure_agreement.compute_agreement(annotations1,annotations1)
    assert(all(value == 1 for value in kappa_dict.values()))
