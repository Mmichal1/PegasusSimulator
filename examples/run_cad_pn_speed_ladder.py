#!/usr/bin/env python3
"""Increase horizontal target/interceptor speed until a stage fails."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parent
ABLATION_SCRIPT = EXAMPLES_DIR / "run_cad_pn_ablation.py"
DEFAULT_LADDER = (
    (0.5, 3.0),
    (1.0, 4.5),
    (2.0, 6.0),
    (3.0, 8.0),
    (4.0, 10.0),
    (5.0, 12.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the straight horizontal CAD chase speed ladder."
    )
    parser.add_argument(
        "--isaac-python",
        default=os.getenv("ISAAC_PYTHON", "/isaac-sim/python.sh"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--required-hit-rate", type=float, default=0.90)
    parser.add_argument(
        "--require-camera",
        action="store_true",
        help="Stop when camera PN fails; otherwise advance using the oracle control envelope.",
    )
    parser.add_argument("--continue-after-failure", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or EXAMPLES_DIR / "logs" / "cad_pn" / f"speed_ladder_{stamp}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_results = []
    failed = False
    for index, (target_speed, pursuer_speed) in enumerate(DEFAULT_LADDER, start=1):
        stage_dir = output_dir / (
            f"stage_{index}_target_{target_speed:g}_pursuer_{pursuer_speed:g}"
        )
        env = os.environ.copy()
        env.update(
            {
                "PEGASUS_ENGAGEMENT": "horizontal",
                "PEGASUS_TARGET_MOTION": "straight",
                "PEGASUS_TARGET_SPEED_MPS": str(target_speed),
                "PEGASUS_PURSUER_SPEED_MPS": str(pursuer_speed),
            }
        )
        command = [
            "python3",
            str(ABLATION_SCRIPT),
            "--isaac-python",
            args.isaac_python,
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--modes",
            "oracle_pn",
            "camera_pn",
            "--motions",
            "straight",
            "--duration",
            str(args.duration),
            "--required-camera-hit-rate",
            "0",
            "--continue-on-error",
            "--output-dir",
            str(stage_dir),
        ]
        print(
            f"Stage {index}/{len(DEFAULT_LADDER)}: target={target_speed:g} m/s, "
            f"interceptor={pursuer_speed:g} m/s",
            flush=True,
        )
        completed = subprocess.run(
            command, cwd=EXAMPLES_DIR.parent, env=env, check=False
        )
        report_path = stage_dir / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            camera_metrics = report["groups"].get("camera_pn/straight", {})
            oracle_metrics = report["groups"].get("oracle_pn/straight", {})
        else:
            camera_metrics = {}
            oracle_metrics = {}
        camera_hit_rate = float(camera_metrics.get("hit_rate", 0.0))
        oracle_hit_rate = float(oracle_metrics.get("hit_rate", 0.0))
        envelope_passed = (
            completed.returncode == 0
            and oracle_hit_rate >= args.required_hit_rate
        )
        camera_passed = camera_hit_rate >= args.required_hit_rate
        passed = envelope_passed and (camera_passed or not args.require_camera)
        stage_results.append(
            {
                "stage": index,
                "target_speed_mps": target_speed,
                "pursuer_speed_mps": pursuer_speed,
                "oracle_hit_rate": oracle_hit_rate,
                "camera_hit_rate": camera_hit_rate,
                "envelope_passed": envelope_passed,
                "camera_passed": camera_passed,
                "passed": passed,
                "report": str(report_path),
            }
        )
        if not passed:
            failed = True
            if not args.continue_after_failure:
                break

    ladder_report = output_dir / "speed_ladder_report.json"
    ladder_report.write_text(
        json.dumps({"stages": stage_results}, indent=2), encoding="utf-8"
    )
    print(f"Speed ladder report: {ladder_report}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
