# Detekcja_filtr_09

Cztery samodzielne skrypty detekcji i śledzenia drona, nazwane wprost od
złożoności obliczeniowej. Każdy ładuje modele z `models/`, przetwarza wideo
albo strumień z kamery i podaje przefiltrowany namiar celu.

Folder jest **przenośny** — zawiera modele, bibliotekę i skrypty. Można go
skopiować na pokład bez reszty projektu.

## Warianty

| skrypt | ms/klatkę | FPS | przyspieszenie | detektor | IoU | błąd środka |
|---|---:|---:|---:|---:|---:|---:|
| `01_bardzo_niska_zlozonosc_mosse.py` | **9.5** | 106 | **16.2×** | 13.6 % | 0.646 | 11.4 px |
| `02_niska_zlozonosc_ncc.py` | 14.8 | 74 | 10.5× | 17.4 % | 0.685 | 9.6 px |
| `03_srednia_zlozonosc_nanotrack.py` | 28.6 | 35 | 5.4× | 13.3 % | 0.750 | 10.9 px |
| `04_wysoka_zlozonosc_botsort.py` | 154.6 | 6.5 | 1.0× | 100 % | **0.945** | **3.8 px** |

Pomiar: 3 nagrania × 240 klatek 1280×720, CPU (Apple Silicon), torch 8 wątków.
Odniesieniem jakości jest pełnoklatkowa detekcja YOLO na każdej klatce, więc
wariant 04 — który robi dokładnie to — z definicji wypada najbliżej.

**Czasy bezwzględne zależą od sprzętu.** Na docelowej platformie zmierz sam:

```bash
python porownaj.py --videos ../film2.mp4
```

## Który wybrać

- **Ciasny budżet CPU, procesor ma też inne zadania** → `01`. Najtańszy; MOSSE
  nie estymuje skali, więc rozmiar ramki opiera się na okresowej korekcie detektora.
- **Nie masz powodu wybrać inaczej** → `02`. Dopasowanie wzorca estymuje skalę,
  a przy rzadkiej weryfikacji degraduje się najłagodniej z całej trójki.
- **Jest zapas CPU, liczy się jakość namiaru** → `03`. Jako jedyny utrzymuje cel
  samodzielnie przez długi czas — w teście bez ani jednej detekcji przez 240 klatek
  zachował IoU 0.676, podczas gdy metody klasyczne odpłynęły do 0.34. Wybór przy
  długich przesłonięciach albo zawodnym detektorze.
- **Koszt nie gra roli, albo w kadrze bywa więcej niż jeden dron** → `04`.
  Najlepsza jakość i osobne ścieżki dla wielu obiektów. Przy 24 FPS budżet to
  42 ms/klatkę — ten wariant go nie zmieści na słabszym CPU.

## Uruchomienie

```bash
python 01_bardzo_niska_zlozonosc_mosse.py --source ../film2.mp4 --save wynik.mp4
```

```bash
python 02_niska_zlozonosc_ncc.py --source 0 --show
```

Wspólne opcje wszystkich czterech skryptów:

| opcja | znaczenie |
|---|---|
| `--source` | plik wideo albo indeks kamery (domyślnie `0`) |
| `--save PLIK.mp4` | zapis nagrania z nałożonymi ramkami i HUD |
| `--show` | podgląd na żywo (klawisz `q` kończy) |
| `--csv PLIK.csv` | ślad celu klatka po klatce do dalszej analizy |
| `--conf` | próg pewności detekcji (domyślnie 0.35) |
| `--device` | `cpu` / `mps` / `cuda` / `0` |
| `--threads 1` | ogranicza wątki torch i OpenCV — scenariusz jednordzeniowy |
| `--max-frames` | limit klatek |
| `--no-filter` | **wyłącza warstwę filtracji** — do porównania surowego sygnału |

Warianty 01–03 mają dodatkowo `--verify-every` (co ile klatek detektor sprawdza
cel w ROI — więcej znaczy taniej i mniej dokładnie) oraz `--imgsz-roi`.
Wariant 04 ma `--tracker-config` (`botsort.yaml` / `bytetrack.yaml`) i `--imgsz`.

## Filtracja — wspólna dla wszystkich czterech

Wszystkie warianty, łącznie z BoT-SORT, podają sygnał o tych samych
właściwościach. Surowy bbox nie idzie prosto na wyjście:

| mechanizm | odcina |
|---|---|
| próg pewności + ROI | detekcje słabe i przestrzennie niemożliwe |
| potwierdzenie **M z N** (3 w oknie 5) | jednoklatkowe fałszywe zablokowanie na ptaku |
| bramka Mahalanobisa | pomiary niezgodne z predykcją — detektora **oraz** trackera |
| kontrola wyglądu (HSV + gradient) | ciche przejęcie innego obiektu po przesłonięciu |
| filtr Kalmana | szum pomiaru, ekstrapolacja w klatkach bez detekcji |
| ogranicznik skoku | nadwyżkę ponad przewidziany ruch celu |
| monitor NIS | niespójność samego filtru |

Po pierwszej detekcji tor jest w stanie **WSTĘPNY** — śledzony wewnętrznie, ale
**na zewnątrz nie wychodzi żaden centroid**, dopóki detektor nie potwierdzi celu
3 razy. Kosztuje to 2–3 klatki przy przechwyceniu.

`--no-filter` wyłącza całą warstwę, jeśli chcesz zobaczyć różnicę.

## Co czyta układ naprowadzania

```python
from lib import HybridDroneTracker

tracker = HybridDroneTracker("models/yolo_based_drone_detector.pt", backend="mosse")

st = tracker.step(frame_bgr)
st.centroid       # punkt celowania po ograniczniku — to podawaj do sterowania
st.centroid_raw   # surowa estymata Kalmana, bez wygładzania
st.quality        # łączna wiarygodność [0,1] — używaj JEJ, nie samej obecności centroidu
st.settling       # True = estymata nadgania korektę, zmniejsz wzmocnienie
st.rejected       # powód odrzucenia pomiaru w tej klatce

los = tracker.los(frame.shape)   # uchyby, tempo zmiany linii celowania, proxy odległości
```

`los()` zwraca `valid=False`, gdy `quality` spadnie poniżej `min_quality` (0.2) —
układ nie dostanie namiaru, któremu nie należy ufać.

Interfejs `__call__(rgb) -> (centroid, debug_img)` pozostaje zgodny z
`YoloDroneTracker`, więc podmiana w istniejącym kontrolerze to jedna linijka:

```python
tracker_instance = HybridDroneTracker(
    model_path="models/yolo_based_drone_detector.pt",
    backend="mosse", conf_thresh=0.4,
)
controller = PluggableTorpedoVisionController(detector=tracker_instance, ...)
```

## Zawartość

```
Detekcja_filtr_09/
├── 01_bardzo_niska_zlozonosc_mosse.py      YOLO + MOSSE
├── 02_niska_zlozonosc_ncc.py               YOLO + dopasowanie wzorca
├── 03_srednia_zlozonosc_nanotrack.py       YOLO + NanoTrack v2
├── 04_wysoka_zlozonosc_botsort.py          YOLO + BoT-SORT + filtracja
├── porownaj.py                             pomiar wszystkich czterech na jednym materiale
├── models/
│   ├── yolo_based_drone_detector.pt        YOLOv10n, klasa 'dron', 640 px   (5.5 MB)
│   ├── nanotrack_backbone_sim.onnx         NanoTrack v2 — wariant 03        (1.0 MB)
│   └── nanotrack_head_sim.onnx             NanoTrack v2 — wariant 03        (0.7 MB)
└── lib/                                    biblioteka wspólna (kopia pakietu tracking/)
    ├── backends.py    wymienne trackery pojedynczego obiektu
    ├── botsort.py     BoT-SORT + warstwa filtracji
    ├── detector.py    YOLO z trybem ROI
    ├── filters.py     potwierdzanie locka, wygląd, ogranicznik, monitor NIS
    ├── hybrid.py      maszyna stanów SEARCH/WSTĘPNY/TRACK/COAST
    ├── kalman.py      filtr Kalmana w płaszczyźnie obrazu
    └── runner.py      wspólna obsługa wejścia, podglądu i statystyk
```

Wymagania: `ultralytics`, `opencv-python`, `torch`, `numpy`. Bez GPU.

`lib/` jest kopią pakietu `tracking/` z katalogu nadrzędnego — zmiany
wprowadzone tam nie przeniosą się tutaj automatycznie.

## Ograniczenia

- **Reguła M z N nie została zweryfikowana na prawdziwych fałszywych detekcjach.**
  W każdej klatce nagrań testowych jest prawdziwy dron, więc licznik
  `niepotwierdzony` pozostaje zerowy. Mechanizm działa w teście syntetycznym
  (`../tracking/test_filters.py`), ale realną wartość pokaże dopiero materiał
  z ptakami i fałszywymi alarmami. To samo dotyczy progów `appearance_floor`
  i `confirm_hits` — są rozsądne, nie zoptymalizowane na danych.
- **Deskryptor wyglądu to histogramy, nie ReID.** Rozróżni drona od tła
  (0.87 vs 0.19), ale dwa podobne drony obok siebie go zmylą.
- **Cel w nagraniach testowych jest duży** (mediana boku 328–673 px na 1280×720),
  więc ROI obejmuje większość klatki i zysk pochodzi głównie z rzadszego wołania
  detektora. Przy małym, odległym dronie cięcie ROI dołoży się osobno — warto to
  zmierzyć na własnym materiale przed przyjęciem tych liczb jako docelowych.
- **NIS wariantu 04 wynosi 0.66**, czyli filtr jest tam zachowawczy. To skutek
  podwójnej filtracji: BoT-SORT ma własny filtr Kalmana. Zmniejszenie szumu
  procesowego (`q_vel` 12 → 1) poprawiło to z 0.22 i zbiło maksymalny skok
  z 41 do 31 px; dalsze zbliżanie do 2.0 kosztowałoby dokładność.
