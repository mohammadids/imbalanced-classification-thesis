def test_minority_class_is_remapped_to_one(bundles):
    for bundle in bundles.values():
        assert int((bundle.y == 1).sum()) == bundle.metadata["minority_class_count"]
        assert int((bundle.y == 1).sum()) < int((bundle.y == 0).sum())


def test_majority_class_is_remapped_to_zero(bundles):
    for bundle in bundles.values():
        assert int((bundle.y == 0).sum()) == bundle.metadata["majority_class_count"]
        assert int((bundle.y == 0).sum()) > int((bundle.y == 1).sum())


def test_original_labels_are_preserved_in_metadata(bundles):
    for bundle in bundles.values():
        labels_from_data = set(bundle.original_y.unique().tolist())
        labels_from_metadata = set(bundle.metadata["original_class_labels"])
        assert labels_from_data == labels_from_metadata
        assert bundle.metadata["majority_class_label"] in labels_from_data
        assert bundle.metadata["minority_class_label"] in labels_from_data

