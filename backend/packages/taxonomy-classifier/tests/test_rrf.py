import numpy as np

from taxonomy_classifier.assignment.rrf import ranks, rrf


def test_ranks():
    S = np.array([[0.1, 0.9, 0.5], [3.0, 2.0, 1.0]])
    assert ranks(S).tolist() == [[3, 1, 2], [1, 2, 3]]


def test_rrf_formula_and_order():
    Sd = np.array([[0.9, 0.8, 0.1]])
    Ss = np.array([[0.1, 0.9, 0.8]])
    R = rrf(Sd, Ss, 60)
    assert np.allclose(R[0], [1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62])
    assert int(np.argmax(R[0])) == 1          # consistently high in both rankings wins
