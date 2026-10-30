import numpy as np

from mtah.harmonise import harmonise, merge_short_sandwiches


def test_merge_short_sandwich_relabels_touch():
    labels = np.array(list("HHHHAAHHHH"))
    assert "".join(merge_short_sandwiches(labels, 3)) == "HHHHHHHHHH"
    # Long enough to be a possession: untouched.
    assert "".join(merge_short_sandwiches(labels, 2)) == "HHHHAAHHHH"


def test_merge_leaves_spells_next_to_dead_ball_and_other_team():
    labels = np.array(list("HHDAAHHH"))        # A follows a dead ball: not a sandwich
    assert "".join(merge_short_sandwiches(labels, 5)) == "HHDAAHHH"
    labels = np.array(list("HHAADDHH"))        # A before a dead ball: not a sandwich
    assert "".join(merge_short_sandwiches(labels, 5)) == "HHAADDHH"


def test_merge_is_iterative():
    # Merging the inner A touch exposes an H touch sandwiched between A's.
    labels = np.array(list("AAAAHAHAAAA"))
    assert "".join(merge_short_sandwiches(labels, 2)) == "AAAAAAAAAAA"


def test_harmonise_passes_metrica_through():
    labels = np.array(list("HHAHH"))
    assert "".join(harmonise(labels, "metrica", 50)) == "HHAHH"
    assert "".join(harmonise(labels, "sportec", 50)) == "HHHHH"
