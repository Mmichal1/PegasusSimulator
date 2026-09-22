#!/usr/bin/env python3
"""WARIANT 3 - SREDNIA ZLOZONOSC OBLICZENIOWA   (~29 ms/klatke, ~35 FPS)

Detektor YOLO + siec syjamska NanoTrack v2 + filtr Kalmana + warstwa filtracji.

Jak to dziala
-------------
Zamiast filtru korelacyjnego miedzy detekcjami pracuje lekka siec syjamska
(NanoTrack v2, projektowana pod urzadzenia mobilne). Kosztuje okolo 8 ms na
klatke, ale jako jedyny z przygotowanych trackerow utrzymuje cel samodzielnie
przez dlugi czas: w tescie bez ani jednej detekcji przez 240 klatek zachowal
IoU 0.676, podczas gdy metody klasyczne odplynely do 0.34.

Modele
------
  models/yolo_based_drone_detector.pt     (YOLOv10n, klasa 'dron')
  models/nanotrack_backbone_sim.onnx      (1.0 MB)
  models/nanotrack_head_sim.onnx          (0.7 MB)

Kiedy wybrac
------------
Gdy jakosc namiaru jest wazniejsza od kosztu i jest zapas CPU. Szczegolnie
przy dlugich przeslonieciach albo gdy detektor bywa zawodny - to jedyny
wariant, ktory przetrwa dluzszy okres bez detekcji.

Zmierzone (3 nagrania x 240 klatek, CPU, wzgledem pseudo-prawdy z detektora)
---------------------------------------------------------------------------
  koszt        28.6 ms/klatke      przyspieszenie   5.4x wobec wariantu 4
  detektor     13.3% klatek        lock            99.2% klatek
  IoU           0.750              blad srodka     10.9 px
  p95 skoku    24.9 px             MAX skok        37.6 px
  NIS           1.64 (cel 2.0)

Uruchomienie
------------
  python 03_srednia_zlozonosc_nanotrack.py --source ../film2.mp4 --save wynik.mp4
  python 03_srednia_zlozonosc_nanotrack.py --source 0 --show --verify-every 4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib import HybridDroneTracker  # noqa: E402
from lib.backends import NanoSOT  # noqa: E402
from lib.runner import apply_threads, base_parser, filter_off_kwargs, run  # noqa: E402

NAZWA = "03 | srednia zlozonosc | YOLO + NanoTrack v2"


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--verify-every", type=int, default=8,
                    help="co ile klatek detektor sprawdza cel w ROI (wiecej = taniej)")
    ap.add_argument("--imgsz-roi", type=int, default=256, help="rozdzielczosc detekcji w ROI")
    args = ap.parse_args()
    apply_threads(args.threads)

    ok, msg = NanoSOT.available()
    if not ok:
        raise SystemExit(
            f"NanoTrack niedostepny: {msg}\n"
            "Sprawdz, czy w models/ leza nanotrack_backbone_sim.onnx "
            "i nanotrack_head_sim.onnx."
        )

    kw = filter_off_kwargs() if args.no_filter else {}
    tracker = HybridDroneTracker(
        model_path=args.weights,
        backend="nanotrack",          # siec syjamska ONNX, ~8 ms/klatke
        conf_thresh=args.conf,
        device=args.device,
        verify_every=args.verify_every,
        imgsz_roi=args.imgsz_roi,
        input_is_rgb=False,
        draw=True,
        **kw,
    )
    run(tracker, args, NAZWA)


if __name__ == "__main__":
    main()
