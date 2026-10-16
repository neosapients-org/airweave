"""A wave's new topics have no family; they are placed with their nearest existing topic."""

import pytest

np = pytest.importorskip("numpy")

from airweave.domains.taxonomy.jobs.gateway import place_new_topics_in_families  # noqa: E402


def _final():
    return {
        "broad_families": [
            {"broad_family_id": "F1", "name": "Commercial"},
            {"broad_family_id": "F2", "name": "Risk"},
        ],
        "categories": [
            {"category_id": "A", "broad_family_id": "F1", "broad_family": "Commercial"},
            {"category_id": "B", "broad_family_id": "F2", "broad_family": "Risk"},
            {"category_id": "CAT_NEW_001", "broad_family_id": None, "broad_family": None},
        ],
    }


def test_new_topic_joins_the_family_of_its_nearest_topic():
    final = _final()
    centroids = np.array([[1.0, 0.0], [0.0, 1.0], [0.1, 0.9]])
    assert place_new_topics_in_families(final, centroids) is True
    new = final["categories"][2]
    assert (new["broad_family_id"], new["broad_family"]) == ("F2", "Risk")


def test_nothing_changes_when_every_topic_has_a_family():
    final = _final()
    final["categories"][2].update(broad_family_id="F1", broad_family="Commercial")
    assert place_new_topics_in_families(final, np.eye(3)) is False


def test_mismatched_centroids_or_no_homed_topic_is_refused():
    with pytest.raises(ValueError, match="do not match"):
        place_new_topics_in_families(_final(), np.eye(2))
    final = _final()
    for c in final["categories"]:
        c["broad_family_id"] = None
    with pytest.raises(ValueError, match="no existing topic"):
        place_new_topics_in_families(final, np.eye(3))
