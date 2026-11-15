import numpy as np

from mtah.decode import (
    argmax_labels,
    decode,
    detect_kickoff_frame,
    merge_short_spells,
    viterbi_block,
)
from mtah.io import Tracking


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


def test_detect_kickoff_frame_from_ball_strike_and_fallback():
    n = 200
    ball = np.full((n, 3), np.nan)
    players = np.full((n, 2, 3), np.nan)
    # Ball sits stationary at centre spot (52.65, 34.0) from frame 50..89, struck at frame 90.
    ball[50:90, 0] = 52.65
    ball[50:90, 1] = 34.0
    for f in range(90, 130):
        ball[f, 0] = 52.65 + (f - 89) * 0.4  # 10 m/s
        ball[f, 1] = 34.0
    tr = Tracking(
        ball=ball,
        players=players,
        team_of_slot=np.array([0, 1], dtype=np.int8),
        fps=25.0,
        pitch=(105.3, 68.0),
        game="SYNTH",
        source="synthetic",
        blocks=np.array([[1, 10, 190]]),
    )
    assert detect_kickoff_frame(tr, 40, 140) == 90
    # Unobserved window falls back to midpoint.
    assert detect_kickoff_frame(tr, 150, 190) == 170
