#!/usr/bin/env python3
"""WARIANT 1 - BARDZO NISKA ZLOZONOSC OBLICZENIOWA   (~9.5 ms/klatke, ~106 FPS)

Detektor YOLO + filtr korelacyjny MOSSE + filtr Kalmana + warstwa filtracji.

Jak to dziala
-------------
Detektor uruchamia sie tylko na ~14% klatek - przy przechwyceniu celu oraz
co osma klatke w malym wycinku wokol pozycji przewidzianej przez Kalmana.
Miedzy detekcjami pracuje MOSSE: jedno FFT okna 64x64, okolo 0.5 ms na klatke.

Modele
------
  models/yolo_based_drone_detector.pt     (YOLOv10n, klasa 'dron')
Zadnych dodatkowych wag - MOSSE to czysty numpy.

Kiedy wybrac
------------
Gdy budzet CPU jest ciasny albo procesor ma inne zadania (sterowanie, lacznosc).
Najtansze z przygotowanych rozwiazan, ale tez najmniej dokladne pod wzgledem
rozmiaru ramki - MOSSE nie estymuje skali samodzielnie i polega na okresowej
korekcie z detektora.

Zmierzone (3 nagrania x 240 klatek, CPU, wzgledem pseudo-prawdy z detektora)
---------------------------------------------------------------------------
  koszt         9.5 ms/klatke      przyspieszenie  16.2x wobec wariantu 4
  detektor     13.6% klatek        lock            99.2% klatek
  IoU           0.646              blad srodka     11.4 px
  p95 skoku    22.9 px             MAX skok        44.4 px
  NIS           1.95 (cel 2.0)

Uruchomienie
------------
  python 01_bardzo_niska_zlozonosc_mosse.py --source ../film2.mp4 --save wynik.mp4
  python 01_bardzo_niska_zlozonosc_mosse.py --source 0 --show
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib import HybridDroneTracker  # noqa: E402
from lib.runner import apply_threads, base_parser, filter_off_kwargs, run  # noqa: E402

NAZWA = "01 | bardzo niska zlozonosc | YOLO + MOSSE"


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
        backend="mosse",              # filtr korelacyjny, ~0.5 ms/klatke
        conf_thresh=args.conf,
        device=args.device,
        verify_every=args.verify_every,
        imgsz_roi=args.imgsz_roi,
        input_is_rgb=False,           # cv2 podaje BGR
        draw=True,
        **kw,
    )
    run(tracker, args, NAZWA)


if __name__ == "__main__":
    main()
