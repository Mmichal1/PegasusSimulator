"""Opakowanie detektora YOLO z trybem ROI.

Kluczowa obserwacja kosztowa: inferencja YOLO skaluje sie mniej wiecej
kwadratowo z bokiem wejscia. Detekcja calej klatki przy 640 px kosztuje
~45 ms na CPU (M-series), ale detekcja w wycinku 256 px wokol przewidzianej
pozycji celu to juz ~8 ms. Skoro sledzimy JEDEN obiekt i mamy jego predykcje
z Kalmana, praktycznie nigdy nie musimy patrzec na cala klatke.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Det:
    xyxy: tuple[float, float, float, float]
    conf: float
    cls: int = 0

    @property
    def xywh(self):
        x0, y0, x1, y1 = self.xyxy
        return (x0, y0, x1 - x0, y1 - y0)

    @property
    def center(self):
        x0, y0, x1, y1 = self.xyxy
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


class YoloDetector:
    """Detektor jednoklasowy z dwoma trybami: pelna klatka i wycinek ROI."""

    def __init__(self, weights: str, conf: float = 0.35, iou: float = 0.5,
                 device: str = "cpu", imgsz_full: int = 640, imgsz_roi: int = 256,
                 half: bool = False, classes=None):
        from ultralytics import YOLO  # import leniwy - benchmark bez YOLO tez dziala

        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz_full = int(imgsz_full)
        self.imgsz_roi = int(imgsz_roi)
        self.half = half
        self.classes = classes
        self.names = self.model.names
        self.t_full = 0.0
        self.t_roi = 0.0
        self.n_full = 0
        self.n_roi = 0

    def _infer(self, img, imgsz, conf=None):
        r = self.model.predict(
            img, imgsz=imgsz, conf=self.conf if conf is None else conf, iou=self.iou,
            device=self.device, half=self.half, classes=self.classes, verbose=False,
        )[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy = b.xyxy.cpu().numpy()
        cf = b.conf.cpu().numpy()
        cl = b.cls.cpu().numpy().astype(int)
        return [Det(tuple(map(float, xyxy[i])), float(cf[i]), int(cl[i])) for i in range(len(cf))]

    def detect_full(self, frame_bgr) -> list[Det]:
        t = time.perf_counter()
        d = self._infer(frame_bgr, self.imgsz_full)
        self.t_full += time.perf_counter() - t
        self.n_full += 1
        return d

    def detect_roi(self, frame_bgr, center, side, conf=None) -> list[Det]:
        """Detekcja w kwadratowym wycinku o boku `side` wokol `center`.
        Wspolrzedne zwracane sa juz w ukladzie pelnej klatki."""
        H, W = frame_bgr.shape[:2]
        cx, cy = center
        side = float(max(side, 32))
        x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
        x1 = int(round(cx + side / 2)); y1 = int(round(cy + side / 2))
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(W, x1), min(H, y1)
        if x1c - x0c < 8 or y1c - y0c < 8:
            return []
        crop = frame_bgr[y0c:y1c, x0c:x1c]
        t = time.perf_counter()
        dets = self._infer(crop, self.imgsz_roi, conf=conf)
        self.t_roi += time.perf_counter() - t
        self.n_roi += 1
        return [Det((d.xyxy[0] + x0c, d.xyxy[1] + y0c, d.xyxy[2] + x0c, d.xyxy[3] + y0c),
                    d.conf, d.cls) for d in dets]

    @property
    def timing(self) -> dict:
        return {
            "full_ms": 1e3 * self.t_full / max(self.n_full, 1),
            "roi_ms": 1e3 * self.t_roi / max(self.n_roi, 1),
            "n_full": self.n_full,
            "n_roi": self.n_roi,
        }

    def reset_timing(self):
        self.t_full = self.t_roi = 0.0
        self.n_full = self.n_roi = 0
