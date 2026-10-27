import numpy as np
import pytest

from mtah.io import AWAY, HOME, Tracking, parse_atd_raw
from mtah.baseline import NOBODY, majority_smooth, nearest_team_codes
from mtah.labels import graded_mask, mask_ungraded, spells


def test_parse_atd_raw_handles_full_and_partial_nan(tmp_path):
    raw = tmp_path / "atd.txt"
    raw.write_text(
        "1:0.5,0.5,1.0;NaN,NaN,NaN;0.1,0.2,NaN:0.50,0.51,2.0\n"
        "2:0.6,0.5,1.0;0.9,0.9,0.5;NaN,NaN,NaN:NaN,NaN,NaN\n"
    )
    frame_ids, players, ball = parse_atd_raw(raw, n_slots=3)
    assert frame_ids.tolist() == [1, 2]
    assert players.shape == (2, 3, 3) and ball.shape == (2, 3)
    assert np.isnan(players[0, 1]).all()
    assert players[0, 2, 0] == pytest.approx(0.1) and np.isnan(players[0, 2, 2])
    assert np.isnan(ball[1]).all()
    assert ball[0, 2] == pytest.approx(2.0)


def test_parse_atd_raw_rejects_ragged_line(tmp_path):
    raw = tmp_path / "atd.txt"
    raw.write_text("1:0.5,0.5,1.0;NaN,NaN,NaN:0.5,0.5,0.0\n")
    with pytest.raises(ValueError):
        parse_atd_raw(raw, n_slots=3)


def _tracking(ball, players, teams):
    return Tracking(ball=np.asarray(ball, float), players=np.asarray(players, float),
                    team_of_slot=np.asarray(teams, np.int8), fps=25.0, pitch=(105.3, 68.0),
                    game="T", source="t")


def test_nearest_team_codes():
    nan = np.nan
    tr = _tracking(
        ball=[[10, 10, 0], [nan, nan, nan], [50, 30, 0]],
        players=[[[9, 10, 0], [30, 30, 0]],
                 [[9, 10, 0], [30, 30, 0]],
                 [[nan, nan, nan], [nan, nan, nan]]],
        teams=[HOME, AWAY],
    )
    assert nearest_team_codes(tr).tolist() == [HOME, NOBODY, NOBODY]


def test_majority_smooth_ties_to_home_and_nobody_to_dead():
    codes = np.array([HOME, AWAY, NOBODY, NOBODY, AWAY, AWAY], dtype=np.int8)
    assert majority_smooth(codes, 0).tolist() == ["H", "A", "D", "D", "A", "A"]
    assert majority_smooth(codes, 1).tolist() == ["H", "H", "A", "A", "A", "A"]


def test_graded_mask_and_spells():
    mask = graded_mask(199319)
    assert not mask[11184] and mask[11185] and mask[90651] and not mask[90652]
    assert not mask[110807] and mask[110808] and mask[190197] and not mask[190198]
    labels = np.array(list("HHADDAAXH"))
    assert spells(labels) == [("H", 0, 2), ("A", 2, 3), ("A", 5, 7), ("H", 8, 9)]
    masked = mask_ungraded(np.full(199319, "H"))
    assert (masked[~mask] == "X").all() and (masked[mask] == "H").all()
