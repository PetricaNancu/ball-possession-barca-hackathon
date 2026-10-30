import numpy as np

from mtah.features import (CONTROL_CARRY_FRAMES, compute_features, controlled_possession,
                           fill_ball_gaps, swap_labels, swap_teams)
from mtah.io import AWAY, HOME, Tracking


def test_fill_ball_gaps_short_only():
    xy = np.full((40, 2), np.nan)
    xy[:10] = [0.0, 0.0]
    xy[13:20] = [3.0, 0.0]          # 3-frame gap -> filled
    xy[35:] = [10.0, 0.0]           # 15-frame gap -> not filled with max_fill=5
    filled, interp = fill_ball_gaps(xy, blocks=np.array([[1, 0, 39]]), max_fill=5)
    assert interp[10:13].all() and not interp[20:35].any()
    assert np.allclose(filled[10:13, 0], [0.75, 1.5, 2.25])
    assert np.isnan(filled[25, 0])


def test_controlled_possession_ignores_lone_touch():
    n = 200
    far = np.full(n, 30.0)
    d_team = {HOME: far.copy(), AWAY: far.copy()}
    # Home controls from frame 10 (two consecutive home touches); away makes one lone touch
    # at frame 100 with nobody keeping the ball; home touches again at 120.
    events = [(10, HOME, False), (30, HOME, False), (100, AWAY, True), (120, HOME, False),
              (140, HOME, False)]
    state = controlled_possession(events, n, d_team, blocks=np.array([[1, 0, n - 1]]))
    assert (state[:10] == -1).all()
    assert (state[10:] == HOME).all()
    # The same away touch followed by a carry (ball stays within 2 m of an away player) flips.
    d_team[AWAY][100:100 + CONTROL_CARRY_FRAMES] = 1.0
    state = controlled_possession(events, n, d_team, blocks=np.array([[1, 0, n - 1]]))
    assert (state[100:120] == AWAY).all() and (state[120:] == HOME).all()


def _toy_tracking(n=400, seed=0):
    rng = np.random.default_rng(seed)
    players = np.full((n, 6, 3), np.nan)
    for s in range(6):
        players[:, s, 0] = 20 + 10 * s + rng.normal(0, 0.3, n).cumsum() * 0.05
        players[:, s, 1] = 34 + rng.normal(0, 0.3, n).cumsum() * 0.05
    ball = np.full((n, 3), np.nan)
    ball[:, 0] = np.linspace(20, 70, n)
    ball[:, 1] = 34.0
    ball[150:170] = np.nan
    team = np.array([HOME, HOME, HOME, AWAY, AWAY, AWAY], dtype=np.int8)
    return Tracking(ball=ball, players=players, team_of_slot=team, fps=25.0, pitch=(105.0, 68.0),
                    game="toy", source="toy", blocks=np.array([[1, 0, n - 1]]))


def test_swap_teams_is_an_involution_and_mirrors_columns():
    X, names = compute_features(_toy_tracking())
    assert X.shape == (400, len(names))
    Xs = swap_teams(X, names)
    assert np.allclose(swap_teams(Xs, names), X, equal_nan=True)
    assert np.allclose(Xs[:, names.index("d_ball_H1")], X[:, names.index("d_ball_A1")], equal_nan=True)
    assert np.allclose(Xs[:, names.index("d_margin_sgn")], -X[:, names.index("d_margin_sgn")], equal_nan=True)
    y = np.array([0, 1, 2, -1])
    assert swap_labels(y).tolist() == [1, 0, 2, -1]
