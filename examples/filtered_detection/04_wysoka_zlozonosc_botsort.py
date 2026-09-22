#!/usr/bin/env python3
"""WARIANT 4 - WYSOKA ZLOZONOSC OBLICZENIOWA   (~155 ms/klatke, ~6 FPS)

Detektor YOLO na kazdej klatce + BoT-SORT + warstwa filtracji.

Jak to dziala
-------------
Rozwiazanie referencyjne, zgodne z pierwotnym `track.py`: pelna detekcja
1280x720 -> 640 px na KAZDEJ klatce plus asocjacja BoT-SORT z identyfikatorami
sciezek. Roznica wobec oryginalu polega na tym, ze surowy bbox nie idzie prosto
na wyjscie - przechodzi przez te sama warstwe filtracji co warianty 1-3
(potwierdzenie M z N, bramka Mahalanobisa, kontrola wygladu, filtr Kalmana,
ogranicznik skoku), wiec uklad naprowadzania dostaje sygnal o tych samych
wlasnosciach niezaleznie od wybranego wariantu.

Modele
------
  models/yolo_based_drone_detector.pt     (YOLOv10n, klasa 'dron')
Konfiguracja botsort.yaml pochodzi z pakietu ultralytics.

Kiedy wybrac
------------
Gdy koszt nie gra roli, a liczy sie najlepsza jakosc namiaru - analiza
nagran po locie, weryfikacja pozostalych wariantow, praca na mocnym sprzecie.
Jako jedyny utrzymuje osobne sciezki dla wielu obiektow naraz, wiec ma sens
takze wtedy, gdy w kadrze bywa wiecej niz jeden dron.

Zmierzone (3 nagrania x 240 klatek, CPU, wzgledem pseudo-prawdy z detektora)
---------------------------------------------------------------------------
  koszt       154.6 ms/klatke      przyspieszenie   1.0x (punkt odniesienia)
  detektor    100.0% klatek        lock            99.2% klatek
  IoU           0.945              blad srodka      3.8 px
  p95 skoku    20.1 px             MAX skok        30.9 px
  NIS           0.66 (filtr zachowawczy - BoT-SORT wygladza juz po swojej stronie)

UWAGA: okolo 16x drozszy od wariantu 1. Na pokladzie z ograniczonym CPU
prawdopodobnie nie utrzyma tempa kamery - przy 24 FPS budzet to 42 ms/klatke.

Uruchomienie
------------
  python 04_wysoka_zlozonosc_botsort.py --source ../film2.mp4 --save wynik.mp4
  python 04_wysoka_zlozonosc_botsort.py --source ../film2.mp4 --device mps
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib import BotSortFilteredTracker  # noqa: E402
from lib.runner import apply_threads, base_parser, filter_off_kwargs, run  # noqa: E402

NAZWA = "04 | wysoka zlozonosc | YOLO + BoT-SORT + filtracja"


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--tracker-config", default="botsort.yaml",
                    help="konfiguracja trackera ultralytics (botsort.yaml / bytetrack.yaml)")
    ap.add_argument("--imgsz", type=int, default=640, help="rozdzielczosc detekcji")
    args = ap.parse_args()
    apply_threads(args.threads)

    kw = filter_off_kwargs() if args.no_filter else {}
    tracker = BotSortFilteredTracker(
        model_path=args.weights,
        tracker_config=args.tracker_config,
        conf_thresh=args.conf,
        device=args.device,
        imgsz=args.imgsz,
        input_is_rgb=False,
        draw=True,
        **kw,
    )
    run(tracker, args, NAZWA)


if __name__ == "__main__":
    main()
