"""Hybrydowy tracker drona: detektor + lekki SOT + Kalman + warstwa filtracji.

Idea
----
Obecne podejscie (`track.py`) uruchamia YOLO na KAZDEJ klatce i dokleja do tego
BoT-SORT, czyli tracker wielu obiektow z asocjacja i ReID. Dla zadania
"trzymaj sie jednego celu" to marnotrawstwo: placimy pelny koszt detekcji
1280x720 -> 640 px plus koszt asocjacji, zeby ostatecznie uzyc jednego bboxa.

Maszyna stanow:

    SEARCH    - nie mamy celu. Pelna detekcja co `search_stride` klatek.
                To jedyny moment, w ktorym patrzymy na cala klatke.
    TENTATIVE - cos znaleziono, ale to jeszcze NIE jest lock. Tor jest
                sledzony wewnetrznie i weryfikowany detektorem co klatke,
                dopoki nie zbierze `confirm_hits` potwierdzen w oknie
                `confirm_window`. Na zewnatrz nie wychodzi zaden centroid.
    TRACK     - cel potwierdzony. Co klatke dziala tylko tani SOT.
                Detektor wraca wylacznie w wycinku ROI wokol predykcji Kalmana
                i tylko wtedy, gdy minelo `verify_every` klatek ALBO pewnosc
                SOT spadla ponizej progu (weryfikacja adaptacyjna).
    COAST     - chwilowa utrata. Kalman ekstrapoluje, ROI rosnie z kazda
                klatka. Ponowne przejecie wymaga zgodnosci wygladu.

Filtracja sygnalu wyjsciowego
----------------------------
Detekcja bez filtracji nie nadaje sie do petli naprowadzania. Warstwy, ktore
odcinaja kolejne rodzaje szumu (szczegoly i uzasadnienie w `filters.py`):

    prog pewnosci + ROI  -> odrzuca detekcje slabe i przestrzennie niemozliwe
    potwierdzenie M z N  -> odrzuca jednoklatkowe falszywe zablokowania
    bramka Mahalanobisa  -> odrzuca pomiary niezgodne z predykcja (detektor
                            ORAZ tracker - dryfujacy SOT nie trafia do filtru)
    kontrola wygladu     -> odrzuca ciche przejscie na inny obiekt
    filtr Kalmana        -> wygladza i ekstrapoluje
    ogranicznik skoku    -> zamienia przeskok estymaty na gladki dojazd
                            z flaga `settling` dla regulatora
    monitor NIS          -> pilnuje, czy filtr jest spojny z danymi

Interfejs `__call__(rgb_image) -> (centroid, debug_image)` jest zgodny
z `YoloDroneTracker` z track.py, wiec podmiana w kontrolerze to jedna linijka.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

try:  # dziala i jako pakiet, i jako luzne pliki
    from .backends import create_backend, iou_xywh
    from .detector import Det, YoloDetector
    from .filters import AppearanceModel, LockConfirmer, NISMonitor, OutputLimiter
    from .kalman import BoxKalman
except ImportError:  # pragma: no cover
    from backends import create_backend, iou_xywh
    from detector import Det, YoloDetector
    from filters import AppearanceModel, LockConfirmer, NISMonitor, OutputLimiter
    from kalman import BoxKalman

SEARCH, TENTATIVE, TRACK, COAST = "SEARCH", "TENTATIVE", "TRACK", "COAST"

_STATE_COLOR = {
    SEARCH: (0, 165, 255),
    TENTATIVE: (0, 255, 255),
    TRACK: (0, 230, 0),
    COAST: (0, 220, 255),
}


@dataclass
class TrackState:
    frame_idx: int
    state: str
    locked: bool
    centroid: tuple[float, float] | None        # po ograniczniku - dla regulatora
    centroid_raw: tuple[float, float] | None    # surowe wyjscie filtru Kalmana
    bbox_xywh: tuple[float, float, float, float] | None
    velocity: tuple[float, float]
    confidence: float
    quality: float                              # laczna wiarygodnosc w [0, 1]
    source: str                                 # 'detector' | 'sot' | 'kalman' | '-'
    settling: bool = False                      # wyjscie nadganie skok estymaty
    appearance: float = 1.0                     # zgodnosc wygladu z modelem celu
    nis: float = float("nan")                   # spojnosc ostatniej korekty
    t_sot_ms: float = 0.0
    t_det_ms: float = 0.0
    t_total_ms: float = 0.0
    detector_calls: int = 0
    roi_side: float = 0.0
    rejected: str = ""                          # powod odrzucenia pomiaru


@dataclass
class LOS:
    """Wielkosci gotowe dla prawa naprowadzania (np. proportional navigation)."""
    valid: bool
    ex: float = 0.0           # uchyb poziomy, znormalizowany do [-1, 1]
    ey: float = 0.0           # uchyb pionowy, znormalizowany do [-1, 1]
    los_rate_x: float = 0.0   # tempo zmiany linii celowania [1/s], znormalizowane
    los_rate_y: float = 0.0
    size_ratio: float = 0.0   # bok bboxa / bok obrazu ~ odwrotnosc odleglosci
    quality: float = 0.0
    settling: bool = False    # regulator powinien zredukowac wzmocnienie


class HybridDroneTracker:
    """Drop-in dla YoloDroneTracker, ale z ulamkowym kosztem obliczeniowym
    i z filtracja sygnalu na potrzeby ukladu naprowadzania."""

    def __init__(
        self,
        model_path: str,
        backend: str = "ncc",
        backend_kwargs: dict | None = None,
        conf_thresh: float = 0.35,
        device: str = "cpu",
        imgsz_full: int = 640,
        imgsz_roi: int = 256,
        # --- polityka uzycia detektora ---
        verify_every: int = 8,
        adaptive_verify: bool = True,
        score_floor: float = 0.35,
        min_verify_gap: int = 2,
        search_stride: int = 2,
        roi_scale: float = 3.0,
        roi_min: int = 96,
        roi_max: int = 640,
        roi_conf: float | None = None,
        # --- polityka utraty celu ---
        max_misses: int = 2,
        max_coast: int = 24,
        coast_detect_every: int = 2,
        reinit_iou: float = 0.5,
        # --- filtracja ---
        confirm_hits: int = 3,          # potwierdzenia detektora wymagane do locka
        confirm_window: int = 5,        # w oknie ilu klatek
        confirm_max_age: int = 12,      # po tylu klatkach porzucamy tor wstepny
        gate_sigma: float = 9.0,        # bramka Mahalanobisa dla detekcji
        gate_sigma_sot: float = 6.0,    # ostrzejsza bramka dla pomiaru z SOT
        appearance_check: bool = True,
        appearance_floor: float = 0.5,        # prog przy ponownym przejeciu (COAST)
        appearance_floor_track: float = 0.35, # lagodniejszy prog w trakcie sledzenia
        appearance_lr: float = 0.05,
        limit_output: bool = True,
        limit_base_px: float = 8.0,     # margines ponad przewidziany ruch celu
        limit_vel_gain: float = 1.0,    # prog = vel_gain*|v| + base_px
        limit_ramp: float = 0.6,        # progresywne nadganianie zaleglosci
        min_quality: float = 0.2,       # ponizej -> los() nie jest wazne
        # --- wejscie / wyjscie ---
        input_is_rgb: bool = True,
        draw: bool = True,
        output_while_coasting: bool = True,
        fps: float = 30.0,
        detector: YoloDetector | None = None,
    ):
        self.detector = detector or YoloDetector(
            model_path, conf=conf_thresh, device=device,
            imgsz_full=imgsz_full, imgsz_roi=imgsz_roi,
        )
        self.backend_name = backend
        self.backend_kwargs = backend_kwargs or {}

        self.conf_thresh = conf_thresh
        self.verify_every = verify_every
        self.adaptive_verify = adaptive_verify
        self.score_floor = score_floor
        self.min_verify_gap = min_verify_gap
        self.search_stride = max(1, search_stride)
        self.roi_scale = roi_scale
        self.roi_min, self.roi_max = roi_min, roi_max
        self.roi_conf = roi_conf if roi_conf is not None else max(0.15, conf_thresh * 0.6)
        self.max_misses = max_misses
        self.max_coast = max_coast
        self.coast_detect_every = max(1, coast_detect_every)
        self.reinit_iou = reinit_iou

        self.gate_sigma = gate_sigma
        self.gate_sigma_sot = gate_sigma_sot
        self.appearance_check = appearance_check
        self.appearance_floor = appearance_floor
        self.appearance_floor_track = appearance_floor_track
        self.min_quality = min_quality

        self.input_is_rgb = input_is_rgb
        self.draw = draw
        self.output_while_coasting = output_while_coasting
        self.fps = fps

        self._confirm_cfg = (confirm_hits, confirm_window, confirm_max_age)
        self._appearance_cfg = appearance_lr
        self._limiter_cfg = (limit_base_px, limit_vel_gain, limit_ramp)
        self.limit_output = limit_output

        self.reset()

    # ------------------------------------------------------------------ stan
    def reset(self) -> None:
        self.state = SEARCH
        self.frame_idx = -1
        self.since_verify = 10 ** 6
        self.since_search = 10 ** 6
        self.misses = 0
        self.coast_frames = 0
        self.bbox = None
        self.confidence = 0.0
        self.quality = 0.0
        self.source = "-"
        self.appearance = 1.0
        self.kf = BoxKalman()
        self.sot = create_backend(self.backend_name, **self.backend_kwargs)
        self.confirmer = LockConfirmer(*self._confirm_cfg)
        self.appearance_model = AppearanceModel(lr=self._appearance_cfg)
        self.limiter = OutputLimiter(*self._limiter_cfg)
        self.nis_monitor = NISMonitor()
        self._last_roi = None
        self.history: list[TrackState] = []
        self.rejects = {"bramka_det": 0, "bramka_sot": 0, "wyglad": 0, "niepotwierdzony": 0}
        self.detector.reset_timing()

    # ------------------------------------------------------------- pomocnicze
    def _roi_side(self, grow: float = 1.0) -> float:
        if self.bbox is not None:
            base = max(self.bbox[2], self.bbox[3])
        else:
            base = self.roi_min / self.roi_scale
        side = self.roi_scale * base * grow
        side += 4.0 * self.kf.pos_std   # niepewnosc Kalmana rozszerza okno
        return float(np.clip(side, self.roi_min, self.roi_max))

    def _pick(self, dets: list[Det], ref_box, frame_bgr, floor: float):
        """Wybor detekcji celu. Kandydaci przechodza kolejno przez bramke
        przestrzenna i kontrole wygladu (`floor` zalezy od stanu - ostrzejszy
        przy ponownym przejeciu po utracie celu); dopiero pozostali sa oceniani
        laczna miara pewnosci, pokrycia, bliskosci i podobienstwa."""
        if not dets:
            return None, ""
        reason = ""
        cands = []
        for d in dets:
            if self.kf.initialised and self.kf.gating_distance(d.xywh, 0.5) > self.gate_sigma:
                self.rejects["bramka_det"] += 1
                reason = "bramka_det"
                continue
            sim = 1.0
            if self.appearance_check and self.appearance_model.desc is not None:
                sim = self.appearance_model.similarity(frame_bgr, d.xywh)
                if sim < floor:
                    self.rejects["wyglad"] += 1
                    reason = "wyglad"
                    continue
            cands.append((d, sim))
        if not cands:
            return None, reason
        if ref_box is None:
            d, sim = max(cands, key=lambda c: c[0].conf)
            self.appearance = sim
            return d, ""
        rcx = ref_box[0] + ref_box[2] / 2.0
        rcy = ref_box[1] + ref_box[3] / 2.0
        diag = float(np.hypot(ref_box[2], ref_box[3])) + 1e-6
        best, best_s, best_sim = None, -1.0, 1.0
        for d, sim in cands:
            ov = iou_xywh(d.xywh, ref_box)
            cx, cy = d.center
            prox = 1.0 / (1.0 + np.hypot(cx - rcx, cy - rcy) / diag)
            s = 0.4 * d.conf + 0.25 * ov + 0.15 * prox + 0.2 * sim
            if s > best_s:
                best, best_s, best_sim = d, s, sim
        self.appearance = best_sim
        return best, ""

    def _start_tentative(self, frame_bgr, det: Det) -> None:
        """Poczatek toru WSTEPNEGO - jeszcze nie lock. Nic nie wychodzi
        na zewnatrz, dopoki detektor nie potwierdzi celu M razy."""
        box = det.xywh
        self.sot = create_backend(self.backend_name, **self.backend_kwargs)
        self.sot.init(frame_bgr, box)
        self.kf.reset(box)
        self.appearance_model = AppearanceModel(lr=self._appearance_cfg)
        self.appearance_model.init(frame_bgr, box)
        self.limiter.reset()
        self.confirmer.reset()
        self.confirmer.push(True)
        self.bbox = box
        self.state = TENTATIVE
        self.misses = 0
        self.coast_frames = 0
        self.since_verify = 0
        self.confidence = det.conf
        self.appearance = 1.0
        self.source = "detector"

    def _drop(self) -> None:
        self.state = SEARCH
        self.bbox = None
        self.confidence = 0.0
        self.quality = 0.0
        self.source = "-"
        self.kf = BoxKalman()
        self.confirmer.reset()
        self.limiter.reset()
        self.since_search = 10 ** 6

    def _compute_quality(self) -> float:
        """Laczna wiarygodnosc wyjscia. Regulator powinien uzywac tej liczby,
        a nie samej obecnosci centroidu."""
        if self.state == SEARCH:
            return 0.0
        q = float(self.confidence)
        if self.state == TENTATIVE:
            q *= 0.0                      # tor niepotwierdzony nie jest wyjsciem
        elif self.state == COAST:
            q *= max(0.0, 1.0 - self.coast_frames / max(self.max_coast, 1)) * 0.6
        if self.appearance_check:
            q *= float(np.clip(self.appearance / max(self.appearance_floor, 1e-3), 0.0, 1.0))
        if self.limiter.settling:
            q *= 0.6                      # estymata jeszcze nie ustabilizowana
        if not self.nis_monitor.consistent and self.nis_monitor.buf:
            q *= 0.8
        return float(np.clip(q, 0.0, 1.0))

    # -------------------------------------------------------------- glowny krok
    def step(self, frame_bgr: np.ndarray) -> TrackState:
        t0 = time.perf_counter()
        self.frame_idx += 1
        self.since_verify += 1
        self.since_search += 1
        t_sot = t_det = 0.0
        det_calls = 0
        roi_side = 0.0
        nis = float("nan")
        rejected = ""
        self._last_roi = None

        if self.kf.initialised:
            self.kf.predict()

        # ---------------------------------------------------------- SEARCH
        if self.state == SEARCH:
            if self.since_search >= self.search_stride:
                self.since_search = 0
                td = time.perf_counter()
                dets = self.detector.detect_full(frame_bgr)
                t_det = (time.perf_counter() - td) * 1e3
                det_calls = 1
                roi_side = float(max(frame_bgr.shape[:2]))
                best = max(dets, key=lambda d: d.conf) if dets else None
                if best is not None:
                    self._start_tentative(frame_bgr, best)
                else:
                    self.bbox = None
                    self.confidence = 0.0
                    self.source = "-"

        # ------------------------------------------- TENTATIVE / TRACK / COAST
        else:
            sot_ok, sot_box, sot_score = False, self.bbox, 0.0
            if self.state in (TENTATIVE, TRACK):
                ts = time.perf_counter()
                sot_ok, sot_box, sot_score = self.sot.update(frame_bgr)
                t_sot = (time.perf_counter() - ts) * 1e3
                if sot_ok:
                    # Bramkowanie pomiaru z trackera: dryfujacy SOT nie moze
                    # wciagnac filtru za soba. Odrzucony pomiar liczy sie jak
                    # porazka SOT, wiec wywola weryfikacje detektorem.
                    if self.kf.gating_distance(sot_box, 3.0) > self.gate_sigma_sot:
                        self.rejects["bramka_sot"] += 1
                        rejected = "bramka_sot"
                        sot_ok = False
                if sot_ok:
                    # NIS z tego pomiaru swiadomie NIE trafia do monitora:
                    # bbox z SOT jest pochodna poprzedniego stanu filtru, wiec
                    # innowacje sa skorelowane, nie biale, i test spojnosci
                    # wychodzilby sztucznie zanizony. Monitor karmimy wylacznie
                    # pomiarami detektora, ktore sa (niemal) niezalezne.
                    self.kf.update(sot_box,
                                   meas_noise_scale=3.0 + 6.0 * (1.0 - sot_score))
                    self.bbox = self.kf.bbox_xywh
                    self.confidence = 0.5 + 0.4 * sot_score
                    self.source = "sot"
                else:
                    self.bbox = self.kf.bbox_xywh
                    self.confidence = 0.35
                    self.source = "kalman"

            # czy siegac po detektor?
            if self.state == TENTATIVE:
                need = True                    # tor wstepny weryfikujemy co klatke
                grow = 1.0
            elif self.state == TRACK:
                need = self.since_verify >= self.verify_every
                if self.adaptive_verify and (not sot_ok or sot_score < self.score_floor):
                    need = need or self.since_verify >= self.min_verify_gap
                grow = 1.0
            else:  # COAST - okno rosnie z kazda klatka bez celu
                need = (self.coast_frames % self.coast_detect_every) == 0
                grow = 1.0 + 0.25 * self.coast_frames
                self.bbox = self.kf.bbox_xywh
                self.confidence = max(0.0, 0.3 * (1.0 - self.coast_frames / self.max_coast))
                self.source = "kalman"

            if need:
                self.since_verify = 0
                roi_side = self._roi_side(grow)
                cx, cy = self.kf.center
                self._last_roi = (cx, cy, roi_side)
                td = time.perf_counter()
                dets = self.detector.detect_roi(frame_bgr, (cx, cy), roi_side,
                                                conf=self.roi_conf)
                t_det = (time.perf_counter() - td) * 1e3
                det_calls = 1
                # Kontrola wygladu jest wymagajaca przy ponownym przejeciu po
                # utracie celu - to wtedy grozi ciche przejscie na inny obiekt.
                # Prog zgodnosci wygladu jest ostrzejszy przy ponownym
                # przejeciu po utracie celu - to wtedy grozi ciche przejscie na
                # inny obiekt. W trakcie normalnego sledzenia prog jest
                # lagodniejszy, zeby nie odrzucac celu zmieniajacego orientacje.
                floor = (self.appearance_floor if self.state == COAST
                         else self.appearance_floor_track)
                best, why = self._pick(dets, self.bbox, frame_bgr, floor)
                rejected = rejected or why

                if self.state == TENTATIVE:
                    self.confirmer.push(best is not None)

                if best is not None:
                    nis = self.kf.update(best.xywh, meas_noise_scale=0.6)
                    self.nis_monitor.push(nis)
                    self.bbox = self.kf.bbox_xywh
                    self.confidence = float(best.conf)
                    self.source = "detector"
                    self.misses = 0
                    if self.state in (TRACK, TENTATIVE):
                        self.appearance_model.update(frame_bgr, best.xywh)
                    if self.state == COAST or not sot_ok or \
                            iou_xywh(sot_box, best.xywh) < self.reinit_iou:
                        self.sot = create_backend(self.backend_name, **self.backend_kwargs)
                        self.sot.init(frame_bgr, best.xywh)
                    if self.state == COAST:
                        self.state = TRACK
                        self.coast_frames = 0
                else:
                    self.misses += 1
                    if self.state == TRACK and self.misses >= self.max_misses:
                        self.state = COAST
                        self.coast_frames = 0

            # --- przejscia z toru wstepnego ---
            if self.state == TENTATIVE:
                if self.confirmer.confirmed:
                    self.state = TRACK
                    self.since_verify = 0
                elif self.confirmer.expired:
                    self.rejects["niepotwierdzony"] += 1
                    rejected = "niepotwierdzony"
                    self._drop()

            if self.state == COAST:
                self.coast_frames += 1
                if self.coast_frames > self.max_coast:
                    self._drop()

        # ------------------------------------------------------------ wyjscie
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
            t_sot_ms=t_sot, t_det_ms=t_det,
            t_total_ms=(time.perf_counter() - t0) * 1e3,
            detector_calls=det_calls, roi_side=roi_side, rejected=rejected,
        )
        self.history.append(st)
        return st

    # ------------------------------------------------------- wyjscie dla PN
    def los(self, frame_shape) -> LOS:
        H, W = frame_shape[:2]
        st_ok = self.state in (TRACK, COAST) and self.bbox is not None
        if not st_ok or self.quality < self.min_quality:
            return LOS(valid=False, quality=self.quality,
                       settling=self.limiter.settling)
        cx, cy = self.limiter.out if (self.limit_output and self.limiter.out) else (
            self.bbox[0] + self.bbox[2] / 2.0, self.bbox[1] + self.bbox[3] / 2.0)
        vx, vy = self.kf.velocity
        return LOS(
            valid=True,
            ex=float((cx - W / 2.0) / (W / 2.0)),
            ey=float((cy - H / 2.0) / (H / 2.0)),
            los_rate_x=float(vx / (W / 2.0) * self.fps),
            los_rate_y=float(vy / (H / 2.0) * self.fps),
            size_ratio=float(max(self.bbox[2], self.bbox[3]) / max(W, H)),
            quality=self.quality,
            settling=self.limiter.settling,
        )

    # ----------------------------------------------------------- wizualizacja
    def render(self, frame_bgr: np.ndarray, st: TrackState) -> np.ndarray:
        img = frame_bgr.copy()
        col = _STATE_COLOR[st.state]
        if self._last_roi is not None:
            cx, cy, side = self._last_roi
            p0 = (int(cx - side / 2), int(cy - side / 2))
            p1 = (int(cx + side / 2), int(cy + side / 2))
            cv2.rectangle(img, p0, p1, (255, 200, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f"ROI {int(side)}px", (p0[0], max(12, p0[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
        if st.bbox_xywh is not None:
            x, y, w, h = [int(round(v)) for v in st.bbox_xywh]
            dashed = st.state == TENTATIVE
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 1 if dashed else 2, cv2.LINE_AA)
            tag = (f"WSTEPNY {self.confirmer.progress}" if dashed
                   else f"dron {st.confidence:.2f} [{st.source}]")
            cv2.putText(img, tag, (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        # surowa estymata vs punkt celowania po ograniczniku
        if st.centroid_raw is not None and st.settling:
            r = tuple(int(round(v)) for v in st.centroid_raw)
            c = tuple(int(round(v)) for v in st.centroid)
            cv2.circle(img, r, 4, (120, 120, 255), 1, cv2.LINE_AA)
            cv2.line(img, r, c, (120, 120, 255), 1, cv2.LINE_AA)
        if st.centroid is not None:
            cx, cy = [int(round(v)) for v in st.centroid]
            cv2.drawMarker(img, (cx, cy), col, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            vx, vy = st.velocity
            cv2.arrowedLine(img, (cx, cy), (int(cx + vx * 6), int(cy + vy * 6)),
                            (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
        hud1 = (f"{self.backend_name} | {st.state} | {st.t_total_ms:5.1f} ms "
                f"({1000.0 / max(st.t_total_ms, 1e-3):5.1f} FPS) | "
                f"sot {st.t_sot_ms:4.1f} det {st.t_det_ms:5.1f}")
        hud2 = (f"jakosc {st.quality:.2f} | wyglad {st.appearance:.2f} | "
                f"NIS {self.nis_monitor.mean:5.2f} "
                f"{'| USTALANIE ' if st.settling else '| '}"
                f"{('| odrzut: ' + st.rejected) if st.rejected else ''}")
        cv2.rectangle(img, (0, 0), (img.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(img, hud1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, hud2, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 220, 255), 1, cv2.LINE_AA)
        return img

    # --------------------------------------------- zgodnosc z YoloDroneTracker
    def __call__(self, rgb_image: np.ndarray):
        """Zgodne z track.YoloDroneTracker: (centroid, debug_image).

        Uwaga: w odroznieniu od oryginalu obraz jest poprawnie konwertowany
        RGB->BGR przed podaniem do YOLO (ultralytics oczekuje BGR), a obraz
        debugowy wraca w tej samej przestrzeni, w ktorej przyszedl.
        """
        img = rgb_image
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if self.input_is_rgb else img
        st = self.step(frame_bgr)
        if self.draw:
            debug = self.render(frame_bgr, st)
            if self.input_is_rgb:
                debug = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        else:
            debug = img.copy()
        centroid = None
        if st.centroid is not None:
            centroid = (int(round(st.centroid[0])), int(round(st.centroid[1])))
        return centroid, debug

    # -------------------------------------------------------------- statystyki
    def summary(self) -> dict:
        if not self.history:
            return {}
        tot = np.array([h.t_total_ms for h in self.history])
        det = np.array([h.detector_calls for h in self.history])
        sot = np.array([h.t_sot_ms for h in self.history])
        dts = np.array([h.t_det_ms for h in self.history])
        locked = np.array([h.locked for h in self.history])
        q = np.array([h.quality for h in self.history])
        return {
            "frames": len(self.history),
            "ms_mean": float(tot.mean()),
            "ms_p95": float(np.percentile(tot, 95)),
            "fps": float(1000.0 / max(tot.mean(), 1e-6)),
            "detector_rate": float(det.mean()),
            "sot_ms_mean": float(sot.mean()),
            "det_ms_when_run": float(dts[det > 0].mean()) if (det > 0).any() else 0.0,
            "locked_rate": float(locked.mean()),
            "quality_mean": float(q.mean()),
            "settling_rate": float(np.mean([h.settling for h in self.history])),
            "nis_mean": self.nis_monitor.mean,
            "nis_verdict": self.nis_monitor.verdict(),
            "rejects": dict(self.rejects),
        }
