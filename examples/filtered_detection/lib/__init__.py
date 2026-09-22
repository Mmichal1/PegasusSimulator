"""Biblioteka wspolna dla skryptow detekcji i sledzenia drona.

Kopia pakietu `tracking/` z katalogu nadrzednego projektu, zebrana tutaj,
zeby folder `Detekcja_filtr_09` byl samodzielny i przenosny.
"""
import os

# Wagi ONNX (NanoTrack) leza we wspolnym `models/`, nie obok kodu.
os.environ.setdefault(
    "DRONE_SOT_WEIGHTS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
)

from .backends import REGISTRY, available_backends, create_backend  # noqa: E402
from .botsort import BotSortFilteredTracker  # noqa: E402
from .detector import YoloDetector  # noqa: E402
from .filters import AppearanceModel, LockConfirmer, NISMonitor, OutputLimiter  # noqa: E402
from .hybrid import COAST, SEARCH, TENTATIVE, TRACK, HybridDroneTracker, LOS, TrackState  # noqa: E402
from .kalman import BoxKalman  # noqa: E402

__all__ = [
    "HybridDroneTracker", "BotSortFilteredTracker", "TrackState", "LOS",
    "SEARCH", "TENTATIVE", "TRACK", "COAST", "YoloDetector", "BoxKalman",
    "create_backend", "available_backends", "REGISTRY",
    "AppearanceModel", "LockConfirmer", "OutputLimiter", "NISMonitor",
]
