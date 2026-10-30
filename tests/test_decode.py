import numpy as np

from mtah.decode import argmax_labels, decode, merge_short_spells, viterbi_block


def test_viterbi_removes_flicker_but_keeps_real_switch():
    # 100 frames of H with a 3-frame A flicker, then 100 frames of A.
    p = np.full((200, 3), 0.05)
    p[:100, 0] = 0.9
    p[40:43] = [0.1, 0.85, 0.05]
    p[100:, 1] = 0.9
    log_p = np.log(p)
    path = viterbi_block(log_p, switch_team=4.0, switch_dead=4.0)
    assert (path[:100] == 0).all() and (path[100:] == 1).all()
    # With no penalty the flicker survives.
    raw = viterbi_block(log_p, 0.0, 0.0)
    assert (raw[40:43] == 1).all()


def test_merge_short_spells_goes_to_longer_neighbour():
    s = np.array([0] * 30 + [1] * 5 + [2] * 10)
    out = merge_short_spells(s, min_frames=8)
    assert (out[:30] == 0).all() and (out[30:35] == 0).all() and (out[35:] == 2).all()
    assert (merge_short_spells(s, 1) == s).all()


def test_decode_masks_outside_blocks():
    p = np.full((50, 3), 1 / 3)
    p[10:30, 2] = 0.9
    blocks = np.array([[1, 10, 29]])
    labels = decode(p, blocks, 2.0, 2.0, 0)
    assert (labels[:10] == "X").all() and (labels[30:] == "X").all()
    assert (labels[10:30] == "D").all()
    assert (argmax_labels(p, blocks)[10:30] == "D").all()
