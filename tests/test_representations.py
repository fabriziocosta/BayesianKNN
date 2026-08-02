import numpy as np

from bayesian_knn.representation import make_representation


def test_representations_have_expected_shapes_and_identity_is_dimension_safe():
    X = np.arange(60, dtype=float).reshape(20, 3)
    for family in ("gaussian", "sparse"):
        representation = make_representation(family, 2, 3).fit(X)
        assert representation.transform(X).shape == (20, 2)

    identity = make_representation("identity", 3, 3).fit(X)
    assert np.array_equal(identity.transform(X), X)
    assert identity.parameters()["n_components"] == 3
