#!/usr/bin/env python3
"""WARIANT 2 - NISKA ZLOZONOSC OBLICZENIOWA   (~15 ms/klatke, ~74 FPS)

Detektor YOLO + dopasowanie wzorca (NCC) + filtr Kalmana + warstwa filtracji.

Jak to dziala
-------------
Jak wariant 1, ale zamiast MOSSE pracuje wieloskalowe dopasowanie wzorca:
jedno `cv2.matchTemplate` w oknie przeszukania, na trzech skalach. Dzieki temu
ramka nadaza za zmiana rozmiaru celu miedzy detekcjami - istotne, gdy dron
zbliza sie lub oddala. Kosztuje to okolo 1 ms na klatke zamiast 0.5 ms.

Modele
------
  models/yolo_based_drone_detector.pt     (YOLOv10n, klasa 'dron')
Zadnych dodatkowych wag.

Kiedy wybrac
------------
Domyslny kompromis. Degraduje sie najlagodniej przy rzadkiej weryfikacji -
przy detektorze na 12% klatek nadal utrzymuje skutecznosc 0.885, podczas gdy
MOSSE spada do 0.751. Jesli nie masz powodu wybrac inaczej, wez ten wariant.

Zmierzone (3 nagrania x 240 klatek, CPU, wzgledem pseudo-prawdy z detektora)
---------------------------------------------------------------------------
  koszt        14.8 ms/klatke      przyspieszenie  10.5x wobec wariantu 4
  detektor     17.4% klatek        lock            99.2% klatek
  IoU           0.685              blad srodka      9.6 px
  p95 skoku    26.3 px             MAX skok        42.8 px
  NIS           1.37 (cel 2.0)

Uruchomienie
------------
  python 02_niska_zlozonosc_ncc.py --source ../film2.mp4 --save wynik.mp4
  python 02_niska_zlozonosc_ncc.py --source 0 --show
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib import HybridDroneTracker  # noqa: E402
from lib.runner import apply_threads, base_parser, filter_off_kwargs, run  # noqa: E402

NAZWA = "02 | niska zlozonosc | YOLO + NCC"


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--verify-every", type=int, default=8,
                    help="co ile klatek detektor sprawdza cel w ROI (wiecej = taniej)")
    ap.add_argument("--imgsz-roi", type=int, default=256, help="rozdzielczosc detekcji w ROI")
    args = ap.parse_args()
    apply_threads(args.threads)

    kw = filter_off_kwargs() if args.no_filter else {}
    tracker = HybridDroneTracker(
        model_path=args.weights,
        backend="ncc",                # dopasowanie wzorca z estymacja skali
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
