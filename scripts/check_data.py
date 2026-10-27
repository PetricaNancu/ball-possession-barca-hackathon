"""Sanity-check the data bundle and print the statistics the rest of the plan relies on.

    python scripts/check_data.py            # tactical
    python scripts/check_data.py --source broadcast

Runs the organisers' alignment checks (ATD ids, Smart Tagging clock, homography rows), parses the
ATD into the canonical cache, and reports ball visibility and tracked-object counts on graded
frames -- the numbers the noise simulator has to reproduce.
"""

import argparse
import time

import _bootstrap  # noqa: F401
import numpy as np

import mtah  # noqa: F401  sets MTAH_DATA_DIR
from mtah import paths
from mtah.io import load_homography, load_smart_tagging, load_tracking
from mtah.labels import graded_mask
from starter import load_atd, load_homography as st_homography, load_smart_tagging as st_tagging


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="tactical", choices=["tactical", "broadcast"])
    parser.add_argument("--no-cache", action="store_true", help="re-parse the raw ATD")
    args = parser.parse_args()

    print(f"data dir : {paths.DATA_DIR}")
    print(f"cache dir: {paths.CACHE_DIR}\n")

    report = load_atd.check_alignment_raw(source=args.source)
    print(f"ATD raw  : {report['n_records']} records, ids {report['frame_id_range']}, "
          f"contiguous={report['contiguous']}, one-per-video-frame={report['agrees']}")

    report = st_homography.check_alignment(source=args.source)
    print(f"homography: {report['n_rows']} rows, one-per-video-frame={report['agrees']}")
    _, err, valid = load_homography(source=args.source)
    print(f"            valid {valid.mean():.1%}, median reprojection error "
          f"{np.median(err[valid]):.3f}")

    if args.source == "tactical":
        tags = load_smart_tagging()
        report = st_tagging.check_time_base(st_tagging.load())
        print(f"smart tags: {len(tags)} instances, kickoffs on the video clock={report['agrees']}")
        print(f"            codes: {tags['code'].value_counts().to_dict()}")
        print(f"            team_side: {tags['team_side'].value_counts(dropna=False).to_dict()}")

    t0 = time.time()
    tr = load_tracking(source=args.source, use_cache=not args.no_cache)
    print(f"\ntracking  : {tr.n_frames} frames x {tr.n_slots} slots, loaded in "
          f"{time.time() - t0:.1f}s  (H={tr.meta.get('home_name', '?')}, "
          f"A={tr.meta.get('away_name', '?')})")
    print(f"            slots per team: H={int((tr.team_of_slot == 0).sum())}, "
          f"A={int((tr.team_of_slot == 1).sum())}")

    present = tr.ball_present
    n_obj = tr.player_present.sum(axis=1)
    print(f"            ball visible: all frames {present.mean():.1%}", end="")
    if args.source == "tactical":
        graded = graded_mask(tr.n_frames)
        print(f", graded frames {present[graded].mean():.1%}")
        print(f"            tracked objects on graded frames: mean {n_obj[graded].mean():.1f}, "
              f"max {n_obj[graded].max()}")
        speed = tr.ball[graded & present, 2]
        print(f"            ball speed on graded frames (m/s): median {np.nanmedian(speed):.2f}, "
              f"p95 {np.nanpercentile(speed, 95):.2f}")
        in_bounds = ((tr.ball[:, 0] >= 0) & (tr.ball[:, 0] <= tr.pitch[0])
                     & (tr.ball[:, 1] >= 0) & (tr.ball[:, 1] <= tr.pitch[1]))
        print(f"            ball out of bounds (visible, graded): "
              f"{(~in_bounds)[graded & present].mean():.1%}")
    else:
        print()


if __name__ == "__main__":
    main()
