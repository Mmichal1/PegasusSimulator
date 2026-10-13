#!/usr/bin/env python3
"""Run deterministic CAD-interceptor ablations in separate Isaac processes.

Example:
    python3 examples/run_cad_pn_ablation.py \
        --isaac-python /path/to/isaac-sim/python.sh --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parent
INTERCEPT_SCRIPT = EXAMPLES_DIR / "18_python_cad_pn_intercept.py"
DEFAULT_MODES = ("pure_pursuit", "camera_pn", "oracle_pn")
DEFAULT_MOTIONS = ("stationary", "straight", "circle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CAD pure pursuit, camera-bearing PN, and oracle PN."
    )
    parser.add_argument(
        "--isaac-python",
        default=os.getenv("ISAAC_PYTHON"),
        help="Isaac Sim python.sh path (or set ISAAC_PYTHON).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=DEFAULT_MODES)
    parser.add_argument(
        "--motions", nargs="+", choices=DEFAULT_MOTIONS, default=DEFAULT_MOTIONS
    )
    parser.add_argument("--duration", type=float, default=22.0)
    parser.add_argument("--detector-device", default="cpu")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--required-camera-hit-rate", type=float, default=0.90)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to examples/logs/cad_pn/ablation_<timestamp>.",
    )
    return parser.parse_args()


def aggregate(results: list[dict]) -> dict:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for result in results:
        buckets[(result["guidance_mode"], result["motion"])].append(result)

    groups = {}
    for (mode, motion), rows in sorted(buckets.items()):
        hits = sum(bool(row["hit"]) for row in rows)
        misses = [
            float(row["closest_approach_m"])
            for row in rows
            if row.get("closest_approach_m") is not None
        ]
        groups[f"{mode}/{motion}"] = {
            "runs": len(rows),
            "hits": hits,
            "hit_rate": hits / len(rows),
            "mean_closest_approach_m": (
                sum(misses) / len(misses) if misses else None
            ),
            "max_closest_approach_m": max(misses) if misses else None,
            "mean_detection_loss_s": (
                sum(float(row["detection_loss_s"]) for row in rows) / len(rows)
            ),
        }
    return groups


def main() -> int:
    args = parse_args()
    if not args.isaac_python:
        raise SystemExit("--isaac-python is required (or set ISAAC_PYTHON)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else EXAMPLES_DIR / "logs" / "cad_pn" / f"ablation_{timestamp}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    failures: list[dict] = []
    total = len(args.modes) * len(args.motions) * len(args.seeds)
    run_index = 0

    for mode in args.modes:
        for motion in args.motions:
            for seed in args.seeds:
                run_index += 1
                stem = f"{mode}_{motion}_seed{seed}"
                summary_path = output_dir / f"{stem}.summary.json"
                telemetry_path = output_dir / f"{stem}.csv"
                stdout_path = output_dir / f"{stem}.log"
                env = os.environ.copy()
                env.update(
                    {
                        "PEGASUS_HEADLESS": "1",
                        "PEGASUS_RECORD": "1" if args.record else "0",
                        "PEGASUS_TARGET_SEED": str(seed),
                        "PEGASUS_TARGET_MOTION": motion,
                        "PEGASUS_GUIDANCE_MODE": mode,
                        "PEGASUS_RUN_DURATION_S": str(args.duration),
                        "PEGASUS_DETECTOR_DEVICE": args.detector_device,
                        "PEGASUS_TELEMETRY": "1",
                        "PEGASUS_TELEMETRY_PATH": str(telemetry_path),
                        "PEGASUS_SUMMARY_PATH": str(summary_path),
                    }
                )
                command = [args.isaac_python, str(INTERCEPT_SCRIPT)]
                print(f"[{run_index}/{total}] {mode} {motion} seed={seed}", flush=True)
                with stdout_path.open("w", encoding="utf-8") as output:
                    completed = subprocess.run(
                        command,
                        cwd=EXAMPLES_DIR.parent,
                        env=env,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )

                if completed.returncode != 0 or not summary_path.exists():
                    failure = {
                        "mode": mode,
                        "motion": motion,
                        "seed": seed,
                        "returncode": completed.returncode,
                        "log": str(stdout_path),
                    }
                    failures.append(failure)
                    print(f"  FAILED: {failure}", file=sys.stderr)
                    if not args.continue_on_error:
                        return completed.returncode or 2
                    continue

                with summary_path.open(encoding="utf-8") as summary_file:
                    results.append(json.load(summary_file))

    groups = aggregate(results)
    report = {
        "results": results,
        "groups": groups,
        "failures": failures,
        "configuration": {
            "seeds": args.seeds,
            "modes": args.modes,
            "motions": args.motions,
            "duration_s": args.duration,
            "required_camera_hit_rate": args.required_camera_hit_rate,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, metrics in groups.items():
        print(
            f"{name}: hits={metrics['hits']}/{metrics['runs']} "
            f"rate={metrics['hit_rate']:.1%} "
            f"mean_miss={metrics['mean_closest_approach_m']}"
        )
    print(f"Report: {report_path}")

    moving_groups = [
        groups.get(f"camera_pn/{motion}")
        for motion in ("straight", "circle")
        if motion in args.motions
    ]
    moving_groups = [group for group in moving_groups if group is not None]
    moving_runs = sum(group["runs"] for group in moving_groups)
    moving_hits = sum(group["hits"] for group in moving_groups)
    moving_rate = moving_hits / moving_runs if moving_runs else 1.0

    stationary_camera = groups.get("camera_pn/stationary")
    stationary_pure = groups.get("pure_pursuit/stationary")
    stationary_regressed = (
        stationary_camera is not None
        and stationary_pure is not None
        and stationary_camera["hit_rate"] < stationary_pure["hit_rate"]
    )
    if moving_rate < args.required_camera_hit_rate or stationary_regressed or failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
