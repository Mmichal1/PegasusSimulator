"""Wspolny mechanizm uruchomieniowy dla wszystkich czterech wariantow.

Skrypty w katalogu nadrzednym roznia sie WYLACZNIE konfiguracja trackera -
cala obsluga wejscia, podgladu, zapisu i statystyk jest tutaj, zeby roznice
miedzy wariantami byly widoczne na pierwszy rzut oka.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")
YOLO_WEIGHTS = os.path.join(MODELS, "yolo_based_drone_detector.pt")


def base_parser(opis: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=opis, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0",
                    help="plik wideo, katalog z obrazami albo indeks kamery (domyslnie 0)")
    ap.add_argument("--weights", default=YOLO_WEIGHTS, help="wagi detektora YOLO")
    ap.add_argument("--conf", type=float, default=0.35, help="prog pewnosci detekcji")
    ap.add_argument("--device", default="cpu", help="cpu / mps / cuda / 0")
    ap.add_argument("--save", default="", help="sciezka wyjsciowego mp4")
    ap.add_argument("--show", action="store_true", help="podglad na zywo (klawisz q konczy)")
    ap.add_argument("--max-frames", type=int, default=0, help="limit klatek (0 = bez limitu)")
    ap.add_argument("--threads", type=int, default=0,
                    help="limit watkow torch/OpenCV (1 = scenariusz jednordzeniowy)")
    ap.add_argument("--no-filter", action="store_true",
                    help="wylacza warstwe filtracji - do porownania surowego sygnalu")
    ap.add_argument("--csv", default="", help="zapis sladu celu do pliku CSV")
    return ap


def apply_threads(n: int) -> None:
    if not n:
        return
    import torch
    torch.set_num_threads(n)
    cv2.setNumThreads(n)
    print(f"# torch = {torch.get_num_threads()} watkow, OpenCV ograniczony "
          f"(dekodowanie FFmpeg ma wlasna pule)")


def filter_off_kwargs() -> dict:
    """Nastawy wylaczajace warstwe filtracji - sygnal idzie prosto z detektora."""
    return dict(limit_output=False, appearance_check=False, confirm_hits=1,
                confirm_window=1, min_quality=0.0)


def run(tracker, args, nazwa: str) -> None:
    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"nie moge otworzyc zrodla: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker.fps = fps

    writer = None
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    csv_f = None
    if args.csv:
        csv_f = open(args.csv, "w")
        csv_f.write("klatka,stan,zrodlo,cx,cy,w,h,vx,vy,pewnosc,jakosc,wyglad,ustalanie\n")

    print(f"\n=== {nazwa} ===")
    print(f"zrodlo: {args.source}  ({W}x{H} @ {fps:.1f} FPS)  urzadzenie: {args.device}")
    if args.no_filter:
        print("UWAGA: warstwa filtracji WYLACZONA (--no-filter)")

    n = 0
    t_wall = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and n >= args.max_frames):
                break
            st = tracker.step(frame)
            if writer is not None or args.show:
                vis = tracker.render(frame, st)
                if writer is not None:
                    writer.write(vis)
                if args.show:
                    cv2.imshow(nazwa, vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if csv_f is not None:
                b = st.bbox_xywh or (float("nan"),) * 4
                c = st.centroid or (float("nan"),) * 2
                csv_f.write(f"{st.frame_idx},{st.state},{st.source},{c[0]:.2f},{c[1]:.2f},"
                            f"{b[2]:.2f},{b[3]:.2f},{st.velocity[0]:.2f},{st.velocity[1]:.2f},"
                            f"{st.confidence:.3f},{st.quality:.3f},{st.appearance:.3f},"
                            f"{int(st.settling)}\n")
            n += 1
    except KeyboardInterrupt:
        print("\nprzerwano")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_f is not None:
            csv_f.close()
        cv2.destroyAllWindows()

    wall = time.perf_counter() - t_wall
    s = tracker.summary()
    if not s:
        print("brak klatek do podsumowania")
        return
    print(f"\nklatki            : {n}  ({wall:.1f}s)")
    print(f"sredni czas/klatke: {s['ms_mean']:.2f} ms   (p95 {s['ms_p95']:.2f} ms)")
    print(f"przepustowosc     : {s['fps']:.1f} FPS")
    if s["sot_ms_mean"] > 0:
        print(f"tani tracker      : {s['sot_ms_mean']:.2f} ms/klatke")
    print(f"detektor          : {s['detector_rate']*100:.1f}% klatek, "
          f"{s['det_ms_when_run']:.1f} ms gdy uruchomiony")
    print(f"lock              : {s['locked_rate']*100:.1f}% klatek, "
          f"srednia jakosc {s['quality_mean']:.2f}")
    print(f"ustalanie wyjscia : {s['settling_rate']*100:.1f}% klatek")
    print(f"spojnosc filtru   : {s['nis_verdict']}")
    print(f"odrzucone pomiary : {s['rejects']}")
    if args.save:
        print(f"zapisano          : {args.save}")
    if args.csv:
        print(f"slad celu         : {args.csv}")
