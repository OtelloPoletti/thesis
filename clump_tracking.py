#!/usr/bin/env python3
"""Runs the finder on the selected snapshots and then tracks a leaf."""
from pathlib import Path
import subprocess
import sys

# ======================== CONFIGURATION TO MODIFY ========================
INPUT_DIR = Path("/home/core/O2/filippobarbani/sims/turb_arms4_k4")
OUTPUT_ROOT = Path("/mnt/O1/otello/risultati/cold_clumps_1050_1080")
FIRST = 1050
LAST = 1080
STEP = 1
START_ID = 57  # clump_id of the accepted leaf in the FIRST catalog

# Finder parameters: keep them identical for all snapshots.
CENTER_KPC = (0, 0, 0)
RADIUS_KPC = 10
MODE = "hierarchical"
DENSITY_STEP = 1.2
MIN_CELLS = 8
MIN_RADIUS_CELLS = 1.0
MAX_CONTOUR_CALLS = 50000
INCLUDE_BOUNDARY = True
RHO_MIN = "1e-24"
LOGT_CENTER = "1.151598"
LOGT_HALFWIDTH = "1.151598"
LABEL_PREFIX = "cold_clumps_box"

# To re-run an already performed analysis, set to True only if you are sure
# you want to delete and recreate the existing snapshot folders.
OVERWRITE = False
# ============================================================================

HERE = Path(__file__).resolve().parent
FINDER = HERE / "cf4_tree.py"
TRACKER = HERE / "track_clump.py"


def main():
    input_dir = INPUT_DIR.expanduser().resolve()
    output_root = OUTPUT_ROOT.expanduser().resolve()
    if STEP <= 0 or LAST < FIRST:
        raise ValueError("Invalid interval: check FIRST, LAST and STEP")
    if not FINDER.is_file() or not TRACKER.is_file():
        raise FileNotFoundError(
            "cf4_tree.py and track_clump.py must be next to this script"
        )

    indices = list(range(FIRST, LAST + 1, STEP))
    rhdfs = [input_dir / f"parthenon.restart.{i:05d}.rhdf" for i in indices]
    missing = [str(p) for p in rhdfs if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing snapshots:\n" + "\n".join(missing))

    output_root.mkdir(parents=True, exist_ok=True)
    result_dirs = []
    for i, rhdf in zip(indices, rhdfs):
        label = f"{LABEL_PREFIX}_{i:05d}"
        out_dir = output_root / label
        if out_dir.exists():
            if not OVERWRITE:
                raise FileExistsError(
                    f"Output already exists: {out_dir}. Change OUTPUT_ROOT "
                    "or set OVERWRITE=True to recreate it."
                )
            import shutil
            shutil.rmtree(out_dir)

        cmd = [
            sys.executable, "-u", str(FINDER),
            "--snapshot", str(rhdf),
            "--center-kpc", *map(str, CENTER_KPC),
            "--radius-kpc", str(RADIUS_KPC),
            "--mode", MODE,
            "--density-step", str(DENSITY_STEP),
            "--min-cells", str(MIN_CELLS),
            "--min-radius-cells", str(MIN_RADIUS_CELLS),
            "--max-contour-calls", str(MAX_CONTOUR_CALLS),
            "--overlay-axes", "none",
            "--max-levels", "500",
        ]
        if INCLUDE_BOUNDARY:
            cmd.append("--include-boundary")
        cmd += [
            "--rho-min", RHO_MIN,
            "--logt-center", LOGT_CENTER,
            "--logt-halfwidth", LOGT_HALFWIDTH,
            "--label", label,
            "--out", str(out_dir),
        ]
        print(f"\n=== Finder snapshot {i:05d}: {rhdf} ===", flush=True)
        subprocess.run(cmd, cwd=HERE, check=True)
        result_dirs.append(str(out_dir))

    history = output_root / "clump_history.csv"
    events = output_root / "clump_events.json"
    cmd = [
        sys.executable, str(TRACKER), *result_dirs,
        "--start-id", str(START_ID),
        "--output", str(history),
        "--events", str(events),
        "--event-overlap", "0.25", #almeno il 25% della massa del clump attuale deriva da un clump che non era quello che stavi seguendo, quindi merger
    ]
    print("\n=== Tracking ===", flush=True)
    subprocess.run(cmd, cwd=HERE, check=True)
    print(f"\nCreated: {history} and {events}")


if __name__ == "__main__":
    main()
