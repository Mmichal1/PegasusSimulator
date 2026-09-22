#!/usr/bin/env python3
"""Porownanie wszystkich czterech wariantow na tym samym materiale.

Uruchom na docelowym sprzecie - bezwzgledne czasy zaleza od platformy,
a wlasnie one decyduja o wyborze wariantu.

    python porownaj.py --videos ../film1.mp4 ../film2.mp4
    python porownaj.py --videos ../film2.mp4 --threads 1

Odniesieniem jakosci jest pelnoklatkowa detekcja YOLO na kazdej klatce
(pseudo-prawda) - liczby mowia, jak blisko jestes wyniku najdrozszej metody.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lib import BotSortFilteredTracker, HybridDroneTracker  # noqa: E402
from lib.backends import NanoSOT, iou_xywh  # noqa: E402
from lib.detector import YoloDetector  # noqa: E402
from lib.runner import YOLO_WEIGHTS, apply_threads  # noqa: E402

CACHE = os.path.join(HERE, "gt_cache")

WARIANTY = [
    ("01 bardzo niska", lambda w, d, c: HybridDroneTracker(
        w, backend="mosse", device=d, conf_thresh=c, input_is_rgb=False, draw=False)),
    ("02 niska", lambda w, d, c: HybridDroneTracker(
        w, backend="ncc", device=d, conf_thresh=c, input_is_rgb=False, draw=False)),
    ("03 srednia", lambda w, d, c: HybridDroneTracker(
        w, backend="nanotrack", device=d, conf_thresh=c, input_is_rgb=False, draw=False)),
    ("04 wysoka", lambda w, d, c: BotSortFilteredTracker(
        w, device=d, conf_thresh=c, input_is_rgb=False, draw=False)),
]


def build_gt(video, weights, device, conf=0.25):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{os.path.basename(video)}.json")
    if os.path.exists(path):
        return json.load(open(path))
    det = YoloDetector(weights, conf=conf, device=device)
    cap = cv2.VideoCapture(video)
    gt, prev = [], None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        ds = det.detect_full(fr)
        if not ds:
            gt.append(None)
            continue
        if prev is None:
            b = max(ds, key=lambda d: d.conf)
        else:
            pcx, pcy = prev[0] + prev[2] / 2, prev[1] + prev[3] / 2
            diag = np.hypot(prev[2], prev[3]) + 1e-6
            b = max(ds, key=lambda d: 0.6 * d.conf + 0.4 / (
                1 + np.hypot(d.center[0] - pcx, d.center[1] - pcy) / diag))
        prev = b.xywh
        gt.append([float(v) for v in prev])
    cap.release()
    json.dump(gt, open(path, "w"))
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=["../film2.mp4"])
    ap.add_argument("--weights", default=YOLO_WEIGHTS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    apply_threads(args.threads)

    if not NanoSOT.available()[0]:
        print(f"! wariant 03 pominiety: {NanoSOT.available()[1]}")

    wyniki = {}
    for video in args.videos:
        if not os.path.exists(video):
            print(f"! brak {video}")
            continue
        print(f"\n=== {os.path.basename(video)} ===")
        gt = build_gt(video, args.weights, args.device, 0.25)
        cap = cv2.VideoCapture(video)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        gt = gt[:len(frames)]

        print(f"{'wariant':<17}{'ms':>8}{'FPS':>8}{'det%':>7}{'lock%':>7}"
              f"{'IoU':>7}{'blad sr':>10}{'p95 skok':>10}{'MAX skok':>10}{'NIS':>7}")
        print("-" * 98)
        for nazwa, make in WARIANTY:
            if nazwa.startswith("03") and not NanoSOT.available()[0]:
                continue
            tr = make(args.weights, args.device, args.conf)
            pred, cents = [], []
            for f in frames:
                st = tr.step(f)
                pred.append(st.bbox_xywh if st.locked else None)
                cents.append(st.centroid)
            ious, ce = [], []
            for p, g in zip(pred, gt):
                if not (p and g):
                    continue
                ious.append(iou_xywh(p, g))
                ce.append(np.hypot(p[0] + p[2] / 2 - (g[0] + g[2] / 2),
                                   p[1] + p[3] / 2 - (g[1] + g[3] / 2)))
            j = np.array([np.hypot(b[0] - a[0], b[1] - a[1])
                          for a, b in zip(cents[:-1], cents[1:]) if a and b])
            s = tr.summary()
            row = (s["ms_mean"], s["fps"], s["detector_rate"] * 100, s["locked_rate"] * 100,
                   float(np.mean(ious)) if ious else 0.0,
                   float(np.median(ce)) if ce else float("nan"),
                   float(np.percentile(j, 95)) if len(j) else 0.0,
                   float(j.max()) if len(j) else 0.0, s["nis_mean"])
            wyniki.setdefault(nazwa, []).append(row)
            print(f"{nazwa:<17}{row[0]:8.2f}{row[1]:8.1f}{row[2]:7.1f}{row[3]:7.1f}"
                  f"{row[4]:7.3f}{row[5]:8.1f}px{row[6]:10.1f}{row[7]:10.1f}{row[8]:7.2f}")

    if len(args.videos) > 1 and wyniki:
        print("\n" + "=" * 98)
        print("SREDNIA PO NAGRANIACH")
        print("=" * 98)
        print(f"{'wariant':<17}{'ms':>8}{'FPS':>8}{'det%':>7}{'lock%':>7}"
              f"{'IoU':>7}{'blad sr':>10}{'p95 skok':>10}{'MAX skok':>10}{'NIS':>7}")
        print("-" * 98)
        base = None
        for nazwa, rows in wyniki.items():
            a = np.array(rows).mean(0)
            if nazwa.startswith("04"):
                base = a[0]
            print(f"{nazwa:<17}{a[0]:8.2f}{a[1]:8.1f}{a[2]:7.1f}{a[3]:7.1f}"
                  f"{a[4]:7.3f}{a[5]:8.1f}px{a[6]:10.1f}{a[7]:10.1f}{a[8]:7.2f}")
        if base:
            print("\nprzyspieszenie wzgledem wariantu 04:")
            for nazwa, rows in wyniki.items():
                a = np.array(rows).mean(0)
                print(f"  {nazwa:<17} {base / max(a[0], 1e-6):5.1f}x")


if __name__ == "__main__":
    main()
