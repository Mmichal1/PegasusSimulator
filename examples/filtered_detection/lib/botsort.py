"""YOLO + BoT-SORT z dolozona warstwa filtracji sygnalu.

Wariant referencyjny: detektor pracuje na KAZDEJ klatce, a asocjacje prowadzi
BoT-SORT z ultralytics. To najdrozsze rozwiazanie z przygotowanych, ale daje
najlepsza jakosc namiaru i radzi sobie z wieloma obiektami w kadrze, bo BoT-SORT
utrzymuje osobna sciezke dla kazdego z nich.

Roznica wobec oryginalnego `track.py`: surowy bbox z BoT-SORT nie idzie prosto
na wyjscie. Przechodzi przez te sama warstwe filtracji, co warianty hybrydowe -
potwierdzenie M z N, bramke Mahalanobisa, kontrole wygladu, filtr Kalmana
i ogranicznik skoku - wiec uklad naprowadzania dostaje sygnal o tych samych
wlasnosciach, niezaleznie od wybranego wariantu.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

try:
    from .filters import AppearanceModel, LockConfirmer, NISMonitor, OutputLimiter
    from .hybrid import COAST, SEARCH, TENTATIVE, TRACK, LOS, TrackState, _STATE_COLOR
    from .kalman import BoxKalman
except ImportError:  # pragma: no cover
    from filters import AppearanceModel, LockConfirmer, NISMonitor, OutputLimiter
    from hybrid import COAST, SEARCH, TENTATIVE, TRACK, LOS, TrackState, _STATE_COLOR
    from kalman import BoxKalman


class BotSortFilteredTracker:
    """BoT-SORT jako zrodlo pomiaru + warstwa filtracji na wyjsciu.

    Interfejs identyczny jak `HybridDroneTracker`, wiec skrypty i kontroler
    nie musza wiedziec, ktory wariant jest pod spodem.
    """

    def __init__(
        self,
        model_path: str,
        tracker_config: str = "botsort.yaml",
        conf_thresh: float = 0.35,
        iou: float = 0.5,
        device: str = "cpu",
        imgsz: int = 640,
        # --- filtracja (te same nastawy co w wariantach hybrydowych) ---
        confirm_hits: int = 3,
        confirm_window: int = 5,
        confirm_max_age: int = 12,
        gate_sigma: float = 9.0,
        appearance_check: bool = True,
        appearance_floor: float = 0.5,
        appearance_floor_track: float = 0.35,
        appearance_lr: float = 0.05,
        limit_output: bool = True,
        limit_base_px: float = 8.0,
        limit_vel_gain: float = 1.0,
        limit_ramp: float = 0.6,
        min_quality: float = 0.2,
        # BoT-SORT ma wlasny filtr Kalmana, wiec jego ramki sa juz wygladzone,
        # a korekta przychodzi co klatke (nie co osma, jak w wariantach
        # hybrydowych). Oba te fakty oznaczaja, ze szum procesowy moze byc
        # duzo mniejszy. Zmierzone na film2.mp4: q_vel 12 -> 1 podnosi NIS
        # z 0.22 do 0.73 (blizej docelowych 2.0) i zbija maksymalny skok
        # wyjscia z 41.3 do 31.3 px, kosztem 1 px na bledzie srodka.
        meas_noise_scale: float = 0.3,
        kalman_kwargs: dict | None = None,
        max_coast: int = 24,
        max_misses: int = 2,
        # --- wejscie / wyjscie ---
        input_is_rgb: bool = True,
        draw: bool = True,
        output_while_coasting: bool = True,
        fps: float = 30.0,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.conf_thresh = conf_thresh
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.gate_sigma = gate_sigma
        self.appearance_check = appearance_check
        self.appearance_floor = appearance_floor
        self.appearance_floor_track = appearance_floor_track
        self.min_quality = min_quality
        self.meas_noise_scale = meas_noise_scale
        self._kf_kwargs = kalman_kwargs if kalman_kwargs is not None else dict(
            q_vel=1.0, q_pos=0.1)
        self.max_coast = max_coast
        self.max_misses = max_misses
        self.input_is_rgb = input_is_rgb
        self.draw = draw
        self.output_while_coasting = output_while_coasting
        self.fps = fps
        self.backend_name = "botsort"

        self._confirm_cfg = (confirm_hits, confirm_window, confirm_max_age)
        self._appearance_cfg = appearance_lr
        self._limiter_cfg = (limit_base_px, limit_vel_gain, limit_ramp)
        self.limit_output = limit_output
        self.reset()

    # ------------------------------------------------------------------ stan
    def reset(self) -> None:
        self.state = SEARCH
        self.frame_idx = -1
        self.locked_id = None
        self.misses = 0
        self.coast_frames = 0
        self.bbox = None
        self.confidence = 0.0
        self.quality = 0.0
        self.appearance = 1.0
        self.source = "-"
        self.kf = BoxKalman(**self._kf_kwargs)
        self.confirmer = LockConfirmer(*self._confirm_cfg)
        self.appearance_model = AppearanceModel(lr=self._appearance_cfg)
        self.limiter = OutputLimiter(*self._limiter_cfg)
        self.nis_monitor = NISMonitor()
        self.history: list[TrackState] = []
        self.rejects = {"bramka_det": 0, "bramka_sot": 0, "wyglad": 0, "niepotwierdzony": 0}
        self._last_roi = None
        try:
            self.model.predictor.trackers[0].reset()   # czysty stan BoT-SORT
        except Exception:
            pass

    # ------------------------------------------------------------ pomocnicze
    def _boxes(self, frame_bgr):
        r = self.model.track(source=frame_bgr, tracker=self.tracker_config,
                             conf=self.conf_thresh, iou=self.iou, imgsz=self.imgsz,
                             device=self.device, persist=True, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        ids = b.id.cpu().numpy() if b.id is not None else np.full(len(conf), np.nan)
        out = []
        for i in range(len(conf)):
            x0, y0, x1, y1 = xyxy[i]
            out.append(dict(xywh=(float(x0), float(y0), float(x1 - x0), float(y1 - y0)),
                            conf=float(conf[i]), id=(None if np.isnan(ids[i]) else int(ids[i]))))
        return out

    def _select(self, cands, frame_bgr):
        """Wybor celu. Priorytet ma sciezka, na ktorej jestesmy zablokowani -
        to wlasnie po to BoT-SORT nadaje identyfikatory. Kandydaci przechodza
        bramke przestrzenna i kontrole wygladu; dopiero potem ocene laczna."""
        if not cands:
            return None, ""
        floor = (self.appearance_floor if self.state == COAST
                 else self.appearance_floor_track)
        reason = ""
        keep = []
        for c in cands:
            if self.kf.initialised and self.kf.gating_distance(c["xywh"], 0.5) > self.gate_sigma:
                self.rejects["bramka_det"] += 1
                reason = "bramka_det"
                continue
            sim = 1.0
            if self.appearance_check and self.appearance_model.desc is not None:
                sim = self.appearance_model.similarity(frame_bgr, c["xywh"])
                if sim < floor:
                    self.rejects["wyglad"] += 1
                    reason = "wyglad"
                    continue
            c["sim"] = sim
            keep.append(c)
        if not keep:
            return None, reason
        # sciezka zablokowana ma pierwszenstwo, o ile przeszla kontrole
        if self.locked_id is not None:
            same = [c for c in keep if c["id"] == self.locked_id]
            if same:
                self.appearance = same[0]["sim"]
                return same[0], ""
        if self.bbox is None:
            best = max(keep, key=lambda c: c["conf"])
            self.appearance = best["sim"]
            return best, ""
        rcx = self.bbox[0] + self.bbox[2] / 2.0
        rcy = self.bbox[1] + self.bbox[3] / 2.0
        diag = float(np.hypot(self.bbox[2], self.bbox[3])) + 1e-6
        best, best_s = None, -1.0
        for c in keep:
            x, y, w, h = c["xywh"]
            prox = 1.0 / (1.0 + np.hypot(x + w / 2 - rcx, y + h / 2 - rcy) / diag)
            s = 0.5 * c["conf"] + 0.3 * prox + 0.2 * c["sim"]
            if s > best_s:
                best, best_s = c, s
        self.appearance = best["sim"] if best else 1.0
        return best, ""

    def _drop(self) -> None:
        self.state = SEARCH
        self.bbox = None
        self.confidence = 0.0
        self.quality = 0.0
        self.source = "-"
        self.locked_id = None
        self.kf = BoxKalman(**self._kf_kwargs)
        self.confirmer.reset()
        self.limiter.reset()

    def _compute_quality(self) -> float:
        if self.state == SEARCH:
            return 0.0
        q = float(self.confidence)
        if self.state == TENTATIVE:
            q *= 0.0
        elif self.state == COAST:
            q *= max(0.0, 1.0 - self.coast_frames / max(self.max_coast, 1)) * 0.6
        if self.appearance_check:
            q *= float(np.clip(self.appearance / max(self.appearance_floor, 1e-3), 0.0, 1.0))
        if self.limiter.settling:
            q *= 0.6
        if not self.nis_monitor.consistent and self.nis_monitor.buf:
            q *= 0.8
        return float(np.clip(q, 0.0, 1.0))

    # ------------------------------------------------------------- glowny krok
    def step(self, frame_bgr: np.ndarray) -> TrackState:
        t0 = time.perf_counter()
        self.frame_idx += 1
        nis = float("nan")
        rejected = ""

        if self.kf.initialised:
            self.kf.predict()

        td = time.perf_counter()
        cands = self._boxes(frame_bgr)
        t_det = (time.perf_counter() - td) * 1e3

        best, why = self._select(cands, frame_bgr)
        rejected = why

        if best is not None:
            box = best["xywh"]
            if self.state == SEARCH:
                self.kf.reset(box)
                self.appearance_model = AppearanceModel(lr=self._appearance_cfg)
                self.appearance_model.init(frame_bgr, box)
                self.limiter.reset()
                self.confirmer.reset()
                self.confirmer.push(True)
                self.state = TENTATIVE
                self.locked_id = best["id"]
                self.bbox = box
                self.confidence = best["conf"]
                self.appearance = 1.0
            else:
                if self.state == TENTATIVE:
                    self.confirmer.push(True)
                nis = self.kf.update(box, meas_noise_scale=self.meas_noise_scale)
                self.nis_monitor.push(nis)
                self.bbox = self.kf.bbox_xywh
                self.confidence = best["conf"]
                self.appearance_model.update(frame_bgr, box)
                if best["id"] is not None:
                    self.locked_id = best["id"]
                if self.state == COAST:
                    self.state = TRACK
                    self.coast_frames = 0
            self.misses = 0
            self.source = "detector"
        else:
            if self.state == TENTATIVE:
                self.confirmer.push(False)
            if self.state in (TENTATIVE, TRACK):
                self.misses += 1
                if self.misses >= self.max_misses:
                    self.state = COAST
                    self.coast_frames = 0
            if self.state != SEARCH:
                self.bbox = self.kf.bbox_xywh
                self.confidence = max(0.0, 0.3 * (1.0 - self.coast_frames / self.max_coast))
                self.source = "kalman"

        if self.state == TENTATIVE:
            if self.confirmer.confirmed:
                self.state = TRACK
            elif self.confirmer.expired:
                self.rejects["niepotwierdzony"] += 1
                rejected = "niepotwierdzony"
                self._drop()

        if self.state == COAST:
            self.coast_frames += 1
            if self.coast_frames > self.max_coast:
                self._drop()

        locked = self.state == TRACK or (self.state == COAST and self.output_while_coasting)
        centroid_raw = None
        if locked and self.bbox is not None:
            centroid_raw = (self.bbox[0] + self.bbox[2] / 2.0,
                            self.bbox[1] + self.bbox[3] / 2.0)
        if self.limit_output:
            centroid = self.limiter(centroid_raw, self.kf.velocity)
        else:
            centroid = centroid_raw
            if centroid_raw is None:
                self.limiter.reset()
        self.quality = self._compute_quality()

        st = TrackState(
            frame_idx=self.frame_idx, state=self.state, locked=locked,
            centroid=centroid, centroid_raw=centroid_raw,
            bbox_xywh=tuple(self.bbox) if self.bbox is not None else None,
            velocity=self.kf.velocity if self.kf.initialised else (0.0, 0.0),
            confidence=float(self.confidence), quality=self.quality,
            source=self.source, settling=self.limiter.settling,
            appearance=float(self.appearance), nis=nis,
            t_sot_ms=0.0, t_det_ms=t_det,
            t_total_ms=(time.perf_counter() - t0) * 1e3,
            detector_calls=1, roi_side=float(max(frame_bgr.shape[:2])),
            rejected=rejected,
        )
        self.history.append(st)
        return st

    # ----------------------------------------------------- to samo co hybryda
    los = None          # przypisane nizej, zeby nie duplikowac kodu
    render = None
    __call__ = None

    def summary(self) -> dict:
        if not self.history:
            return {}
        tot = np.array([h.t_total_ms for h in self.history])
        return {
            "frames": len(self.history),
            "ms_mean": float(tot.mean()),
            "ms_p95": float(np.percentile(tot, 95)),
            "fps": float(1000.0 / max(tot.mean(), 1e-6)),
            "detector_rate": 1.0,
            "sot_ms_mean": 0.0,
            "det_ms_when_run": float(np.mean([h.t_det_ms for h in self.history])),
            "locked_rate": float(np.mean([h.locked for h in self.history])),
            "quality_mean": float(np.mean([h.quality for h in self.history])),
            "settling_rate": float(np.mean([h.settling for h in self.history])),
            "nis_mean": self.nis_monitor.mean,
            "nis_verdict": self.nis_monitor.verdict(),
            "rejects": dict(self.rejects),
        }


# Wyjscie dla naprowadzania, rysowanie i zgodnosc z YoloDroneTracker sa
# identyczne jak w wariantach hybrydowych - pozyczamy implementacje zamiast
# utrzymywac druga kopie tego samego kodu.
try:
    from .hybrid import HybridDroneTracker as _H
except ImportError:  # pragma: no cover
    from hybrid import HybridDroneTracker as _H

BotSortFilteredTracker.los = _H.los
BotSortFilteredTracker.render = _H.render
BotSortFilteredTracker.__call__ = _H.__call__
