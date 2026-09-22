"""Wymienne backendy sledzenia pojedynczego obiektu (SOT).

Kazdy backend implementuje ten sam, minimalny kontrakt:

    tr = create_backend("mosse")
    tr.init(frame_bgr, (x, y, w, h))
    ok, bbox_xywh, score = tr.update(frame_bgr)

`score` jest znormalizowana miara pewnosci w [0, 1] - pipeline hybrydowy
uzywa jej do decyzji "czy juz czas dołozyc detektor".

Backendy dziela sie na trzy rodziny:
  * filtry korelacyjne / dopasowanie wzorca (MOSSE, NCC)  - ~0.1-1 ms, brak wag
  * przeplyw optyczny (MedianFlow/LK)                     - ~1-3 ms, brak wag
  * lekkie sieci syjamskie (NanoTrack, VitTrack, DaSiamRPN) - 5-25 ms, wagi ONNX
  * OpenCV klasyczne (MIL/KCF/CSRT jesli dostepne w buildzie)

Zadna z metod nie wymaga GPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

# Katalog wag ONNX. Domyslnie obok tego pliku, ale da sie przestawic zmienna
# srodowiskowa - dzieki temu ta sama biblioteka dziala w roznych ukladach
# katalogow (np. gdy wagi leza we wspolnym `models/`) bez kopiowania kodu.
WEIGHTS_DIR = os.environ.get(
    "DRONE_SOT_WEIGHTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"),
)


def set_weights_dir(path: str) -> None:
    """Przestawia katalog wag ONNX w czasie dzialania."""
    global WEIGHTS_DIR
    WEIGHTS_DIR = os.path.abspath(path)


# --------------------------------------------------------------------- utils
def _clip_box(box, W, H, min_side=4):
    x, y, w, h = box
    w = max(float(w), min_side)
    h = max(float(h), min_side)
    x = min(max(float(x), -w + 1), W - 1)
    y = min(max(float(y), -h + 1), H - 1)
    return (x, y, w, h)


def _crop_resize(img, cx, cy, w, h, out_wh):
    """Wycinek wysrodkowany na (cx, cy) z odbiciem na brzegach, przeskalowany."""
    w = max(int(round(w)), 2)
    h = max(int(round(h)), 2)
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    H, W = img.shape[:2]
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x0 + w - W), max(0, y0 + h - H)
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(W, x0 + w), min(H, y0 + h)
    if xs1 <= xs0 or ys1 <= ys0:
        return np.zeros((out_wh[1], out_wh[0]), dtype=img.dtype)
    patch = img[ys0:ys1, xs0:xs1]
    if px0 or py0 or px1 or py1:
        patch = cv2.copyMakeBorder(patch, py0, py1, px0, px1, cv2.BORDER_REPLICATE)
    return cv2.resize(patch, out_wh, interpolation=cv2.INTER_LINEAR)


def _subpixel_peak(resp: np.ndarray, pk) -> tuple[float, float]:
    """Parabolicza interpolacja piku korelacji.

    Bez tego rozdzielczosc pozycji to 1 piksel okna korelacji, co po
    przeskalowaniu z powrotem na obraz daje blad rzedu (bok_celu / bok_okna).
    Dla celu 600 px i okna 64 px to az ~10 px - czyli wiekszosc bledu
    obserwowanego w benchmarku. Fit paraboliczny zbija to kilkukrotnie.
    """
    y, x = int(pk[0]), int(pk[1])
    h, w = resp.shape
    dy = dx = 0.0
    if 0 < x < w - 1:
        l, c, r = float(resp[y, x - 1]), float(resp[y, x]), float(resp[y, x + 1])
        den = l - 2 * c + r
        if abs(den) > 1e-9:
            dx = float(np.clip(0.5 * (l - r) / den, -1.0, 1.0))
    if 0 < y < h - 1:
        u, c, d = float(resp[y - 1, x]), float(resp[y, x]), float(resp[y + 1, x])
        den = u - 2 * c + d
        if abs(den) > 1e-9:
            dy = float(np.clip(0.5 * (u - d) / den, -1.0, 1.0))
    return y + dy, x + dx


def iou_xywh(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


# ------------------------------------------------------------------ interfejs
@dataclass
class SOTInfo:
    name: str
    family: str
    needs_weights: bool = False
    note: str = ""


class BaseSOT:
    info = SOTInfo("base", "abstract")

    def init(self, frame_bgr: np.ndarray, bbox_xywh) -> None:
        raise NotImplementedError

    def update(self, frame_bgr: np.ndarray):
        raise NotImplementedError

    @classmethod
    def available(cls) -> tuple[bool, str]:
        return True, ""


# ============================================================== 1. MOSSE (wlasny)
class MosseSOT(BaseSOT):
    """Minimum Output Sum of Squared Error (Bolme i in., CVPR 2010).

    Czysty numpy + FFT, zero zaleznosci i zero wag. Najtanszy sensowny tracker:
    caly krok to jedno FFT okna ~64x64. Nie estymuje skali samodzielnie
    (opcjonalne przeszukanie kilku skal), wiec w pipeline liczy na okresowa
    korekte z detektora.
    """

    info = SOTInfo("mosse", "filtr korelacyjny", False, "najszybszy, bez skali")

    def __init__(self, win: int = 64, sigma: float = 2.0, lr: float = 0.125,
                 pad: float = 1.6, n_scales: int = 1, scale_step: float = 1.02,
                 psr_floor: float = 4.0, psr_good: float = 12.0, scale_lr: float = 0.3):
        self.win = int(win)
        self.sigma = sigma
        self.lr = lr
        self.pad = pad
        self.n_scales = max(1, int(n_scales))
        self.scale_step = scale_step
        self.psr_floor = psr_floor
        self.psr_good = psr_good
        self.scale_lr = scale_lr
        self._hann = np.outer(np.hanning(self.win), np.hanning(self.win)).astype(np.float32)
        g = self._gauss_target()
        self._G = np.fft.fft2(g)
        self.A = None
        self.B = None

    def _gauss_target(self) -> np.ndarray:
        n = self.win
        ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
        c = (n - 1) / 2.0
        g = np.exp(-((xs - c) ** 2 + (ys - c) ** 2) / (2.0 * self.sigma ** 2))
        return (g / g.sum()).astype(np.float32) * n * n  # normalizacja energii

    def _prep(self, gray_patch: np.ndarray) -> np.ndarray:
        p = np.log(gray_patch.astype(np.float32) + 1.0)
        p = (p - p.mean()) / (p.std() + 1e-5)
        return p * self._hann

    @staticmethod
    def _rand_warp(img: np.ndarray, rng) -> np.ndarray:
        h, w = img.shape
        ang = rng.uniform(-np.pi / 16, np.pi / 16)
        sc = 1.0 + rng.uniform(-0.1, 0.1)
        c, s = np.cos(ang) * sc, np.sin(ang) * sc
        M = np.float32([[c, -s, 0], [s, c, 0]])
        cc = np.float32([w / 2, h / 2])
        M[:, 2] = cc - M[:, :2] @ cc
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def init(self, frame_bgr, bbox_xywh) -> None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        x, y, w, h = bbox_xywh
        self.cx, self.cy = x + w / 2.0, y + h / 2.0
        self.w, self.h = float(w), float(h)
        patch = _crop_resize(gray, self.cx, self.cy, self.w * self.pad, self.h * self.pad,
                             (self.win, self.win))
        base = self._prep(patch)
        rng = np.random.default_rng(0)
        A = np.zeros((self.win, self.win), dtype=np.complex128)
        B = np.zeros_like(A)
        for i in range(8):
            p = base if i == 0 else self._prep(
                np.clip(self._rand_warp(patch.astype(np.float32), rng), 0, 255))
            F = np.fft.fft2(p)
            A += self._G * np.conj(F)
            B += F * np.conj(F)
        self.A, self.B = A, B
        self.psr = self.psr_good

    def _response(self, gray, cx, cy, w, h):
        patch = _crop_resize(gray, cx, cy, w * self.pad, h * self.pad, (self.win, self.win))
        F = np.fft.fft2(self._prep(patch))
        Hc = self.A / (self.B + 1e-5)
        resp = np.real(np.fft.ifft2(F * Hc))
        return resp, F

    @staticmethod
    def _psr(resp, peak_yx):
        peak = resp[peak_yx]
        mask = np.ones_like(resp, dtype=bool)
        y, x = peak_yx
        n = resp.shape[0]
        y0, y1 = max(0, y - 5), min(n, y + 6)
        x0, x1 = max(0, x - 5), min(n, x + 6)
        mask[y0:y1, x0:x1] = False
        side = resp[mask]
        return float((peak - side.mean()) / (side.std() + 1e-6))

    def update(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        best = None
        scales = [self.scale_step ** i for i in
                  range(-(self.n_scales // 2), self.n_scales // 2 + 1)][:self.n_scales]
        for sc in scales:
            w, h = self.w * sc, self.h * sc
            resp, F = self._response(gray, self.cx, self.cy, w, h)
            pk = np.unravel_index(np.argmax(resp), resp.shape)
            psr = self._psr(resp, pk)
            if best is None or psr > best[0]:
                best = (psr, pk, F, w, h)
                best_resp = resp
        psr, pk, F, w, h = best
        py, px = _subpixel_peak(best_resp, pk)
        c = (self.win - 1) / 2.0
        dy = (py - c) / self.win * (h * self.pad)
        dx = (px - c) / self.win * (w * self.pad)
        self.cx += float(dx)
        self.cy += float(dy)
        if self.n_scales > 1:  # tlumienie skali, patrz NccSOT
            self.w += self.scale_lr * (w - self.w)
            self.h += self.scale_lr * (h - self.h)
        self.psr = psr
        ok = psr >= self.psr_floor
        if ok:  # adaptacja filtra tylko gdy sledzenie wiarygodne
            self.A = self.lr * (self._G * np.conj(F)) + (1 - self.lr) * self.A
            self.B = self.lr * (F * np.conj(F)) + (1 - self.lr) * self.B
        H, W = gray.shape[:2]
        box = _clip_box((self.cx - self.w / 2, self.cy - self.h / 2, self.w, self.h), W, H)
        score = float(np.clip((psr - self.psr_floor) / (self.psr_good - self.psr_floor), 0, 1))
        return ok, box, score


# ======================================================== 2. NCC template match
class NccSOT(BaseSOT):
    """Dopasowanie wzorca (znormalizowana korelacja) w malym oknie przeszukania.

    Jedno `cv2.matchTemplate` na kilku skalach. Praktycznie darmowe dla malych
    obiektow, bardzo dobre dla drona na jednolitym niebie; gubi sie przy
    silnej zmianie wygladu.
    """

    info = SOTInfo("ncc", "dopasowanie wzorca", False, "tanie, ma estymacje skali")

    def __init__(self, search: float = 2.2, tmpl_max: int = 64, lr: float = 0.08,
                 scales=(0.95, 1.0, 1.05), score_floor: float = 0.45,
                 scale_lr: float = 0.3):
        self.search = search
        self.tmpl_max = tmpl_max
        self.lr = lr
        self.scales = tuple(scales)
        self.score_floor = score_floor
        # Zmiana skali nie jest stosowana w calosci: wybor argmax po kilku
        # skalach kompounduje sie z klatki na klatke i bez korekty detektorem
        # bbox potrafi eksplodowac albo zapasc sie do zera. Tlumimy jak DSST.
        self.scale_lr = scale_lr

    def _tmpl_size(self, w, h):
        s = min(1.0, self.tmpl_max / max(w, h, 1.0))
        return max(8, int(round(w * s))), max(8, int(round(h * s)))

    def init(self, frame_bgr, bbox_xywh) -> None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        x, y, w, h = bbox_xywh
        self.cx, self.cy = x + w / 2.0, y + h / 2.0
        self.w, self.h = float(w), float(h)
        self.tw, self.th = self._tmpl_size(self.w, self.h)
        self.tmpl = _crop_resize(gray, self.cx, self.cy, self.w, self.h,
                                 (self.tw, self.th)).astype(np.float32)
        self.score = 1.0

    def update(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        H, W = gray.shape[:2]
        sw = self.w * self.search
        sh = self.h * self.search
        # okno przeszukania w skali wzorca
        ow = max(self.tw + 4, int(round(self.tw * self.search)))
        oh = max(self.th + 4, int(round(self.th * self.search)))
        win = _crop_resize(gray, self.cx, self.cy, sw, sh, (ow, oh)).astype(np.float32)
        best = None
        for sc in self.scales:
            tw = max(6, int(round(self.tw * sc)))
            th = max(6, int(round(self.th * sc)))
            if tw >= ow or th >= oh:
                continue
            tm = cv2.resize(self.tmpl, (tw, th), interpolation=cv2.INTER_LINEAR)
            res = cv2.matchTemplate(win, tm, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if best is None or mx > best[0]:
                sy, sx = _subpixel_peak(res, (loc[1], loc[0]))
                best = (mx, (sx, sy), tw, th, sc)
        if best is None:
            return False, (self.cx - self.w / 2, self.cy - self.h / 2, self.w, self.h), 0.0
        mx, loc, tw, th, sc = best
        # srodek trafienia w ukladzie okna -> uklad obrazu
        px = loc[0] + tw / 2.0
        py = loc[1] + th / 2.0
        self.cx = self.cx - sw / 2.0 + px * (sw / ow)
        self.cy = self.cy - sh / 2.0 + py * (sh / oh)
        sc_damped = 1.0 + self.scale_lr * (sc - 1.0)
        self.w *= sc_damped
        self.h *= sc_damped
        ok = mx >= self.score_floor
        if ok:
            new = _crop_resize(gray, self.cx, self.cy, self.w, self.h,
                               (self.tw, self.th)).astype(np.float32)
            self.tmpl = (1 - self.lr) * self.tmpl + self.lr * new
        self.score = float(mx)
        box = _clip_box((self.cx - self.w / 2, self.cy - self.h / 2, self.w, self.h), W, H)
        return ok, box, float(np.clip((mx - self.score_floor) / (1 - self.score_floor), 0, 1))


# ==================================================== 3. Median Flow (LK + FB)
class MedianFlowSOT(BaseSOT):
    """Median Flow (Kalal i in., ICPR 2010): piramidalny Lucas-Kanade + blad
    forward-backward. Estymuje przesuniecie z mediany i skale z mediany
    stosunkow odleglosci par punktow. Sam wykrywa wlasna porazke (rosnacy FB),
    co czyni go dobrym partnerem dla detektora."""

    info = SOTInfo("medianflow", "przeplyw optyczny", False, "wykrywa wlasna porazke")

    def __init__(self, grid: int = 8, fb_max: float = 6.0, win=(15, 15), levels: int = 3,
                 scale_lr: float = 0.3):
        self.grid = grid
        self.fb_max = fb_max
        self.scale_lr = scale_lr
        self.lk = dict(winSize=win, maxLevel=levels,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        self.prev = None

    def init(self, frame_bgr, bbox_xywh) -> None:
        self.prev = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr.copy()
        x, y, w, h = bbox_xywh
        self.box = [float(x), float(y), float(w), float(h)]
        self.score = 1.0

    def _points(self):
        x, y, w, h = self.box
        m = 0.08
        xs = np.linspace(x + m * w, x + (1 - m) * w, self.grid)
        ys = np.linspace(y + m * h, y + (1 - m) * h, self.grid)
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)

    def update(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        H, W = gray.shape[:2]
        p0 = self._points()
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None, **self.lk)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, **self.lk)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_max)
        self.prev = gray
        if good.sum() < 4:
            self.score = 0.0
            return False, tuple(self.box), 0.0
        a = p0.reshape(-1, 2)[good]
        b = p1.reshape(-1, 2)[good]
        d = np.median(b - a, axis=0)
        # skala: mediana stosunkow odleglosci parami (podprobkowane pary)
        n = len(a)
        idx = np.triu_indices(n, 1)
        if len(idx[0]) > 400:
            sel = np.random.default_rng(0).choice(len(idx[0]), 400, replace=False)
            idx = (idx[0][sel], idx[1][sel])
        da = np.linalg.norm(a[idx[0]] - a[idx[1]], axis=1)
        db = np.linalg.norm(b[idx[0]] - b[idx[1]], axis=1)
        m = da > 1e-3
        scale = float(np.median(db[m] / da[m])) if m.any() else 1.0
        scale = float(np.clip(scale, 0.9, 1.1))
        scale = 1.0 + self.scale_lr * (scale - 1.0)   # tlumienie, patrz NccSOT
        x, y, w, h = self.box
        cx, cy = x + w / 2 + d[0], y + h / 2 + d[1]
        w, h = w * scale, h * scale
        self.box = list(_clip_box((cx - w / 2, cy - h / 2, w, h), W, H))
        med_fb = float(np.median(fb[good]))
        self.score = float(np.clip(1.0 - med_fb / self.fb_max, 0, 1)) * min(1.0, good.sum() / 16.0)
        return True, tuple(self.box), self.score


# ============================================== 4. wrappery trackerow OpenCV
class _CvSOT(BaseSOT):
    _ctor = None  # nazwa atrybutu w cv2

    def __init__(self, **kw):
        self._kw = kw
        self.tracker = None
        self.box = (0, 0, 0, 0)

    @classmethod
    def available(cls):
        if not hasattr(cv2, cls._ctor):
            return False, f"cv2.{cls._ctor} niedostepny w tym buildzie OpenCV"
        return True, ""

    def _make(self):
        raise NotImplementedError

    def init(self, frame_bgr, bbox_xywh):
        self.tracker = self._make()
        x, y, w, h = [int(round(v)) for v in bbox_xywh]
        self.tracker.init(frame_bgr, (x, y, max(w, 2), max(h, 2)))
        self.box = (x, y, w, h)

    def update(self, frame_bgr):
        ok, box = self.tracker.update(frame_bgr)
        if ok:
            self.box = tuple(float(v) for v in box)
        score = getattr(self.tracker, "getTrackingScore", lambda: 1.0 if ok else 0.0)()
        return bool(ok), self.box, float(np.clip(score, 0, 1))


class MilSOT(_CvSOT):
    info = SOTInfo("mil", "online boosting (OpenCV)", False, "wolny, bez skali")
    _ctor = "TrackerMIL"

    def _make(self):
        return cv2.TrackerMIL_create()


class KcfSOT(_CvSOT):
    info = SOTInfo("kcf", "filtr korelacyjny (OpenCV)", False, "wymaga opencv-contrib")
    _ctor = "TrackerKCF"

    def _make(self):
        return cv2.TrackerKCF_create()


class CsrtSOT(_CvSOT):
    info = SOTInfo("csrt", "filtr korelacyjny CSR-DCF (OpenCV)", False,
                   "najdokladniejszy klasyczny, wymaga opencv-contrib")
    _ctor = "TrackerCSRT"

    def _make(self):
        return cv2.TrackerCSRT_create()


class NanoSOT(_CvSOT):
    """NanoTrack v2 - lekka siec syjamska projektowana pod urzadzenia mobilne
    (~1.9 MB ONNX lacznie). Realny kompromis SOTA dla CPU/edge."""

    info = SOTInfo("nanotrack", "siec syjamska (ONNX)", True, "~1.9 MB wag")
    _ctor = "TrackerNano"
    BACKBONE = "nanotrack_backbone_sim.onnx"
    HEAD = "nanotrack_head_sim.onnx"

    @classmethod
    def available(cls):
        ok, msg = super().available()
        if not ok:
            return ok, msg
        for f in (cls.BACKBONE, cls.HEAD):
            if not os.path.exists(os.path.join(WEIGHTS_DIR, f)):
                return False, f"brak wag {f} (uruchom tracking/fetch_weights.py)"
        return True, ""

    def _make(self):
        p = cv2.TrackerNano_Params()
        p.backbone = os.path.join(WEIGHTS_DIR, self.BACKBONE)
        p.neckhead = os.path.join(WEIGHTS_DIR, self.HEAD)
        return cv2.TrackerNano_create(p)


class VitSOT(_CvSOT):
    """VitTrack - lekki tracker transformerowy z OpenCV Zoo (~7 MB ONNX).
    Najlepsza dokladnosc sposrod wariantow "lekkich", kosztem ~10-25 ms/klatke."""

    info = SOTInfo("vittrack", "transformer (ONNX)", True, "~7 MB wag")
    _ctor = "TrackerVit"
    MODEL = "object_tracking_vittrack_2023sep.onnx"

    @classmethod
    def available(cls):
        ok, msg = super().available()
        if not ok:
            return ok, msg
        if not os.path.exists(os.path.join(WEIGHTS_DIR, cls.MODEL)):
            return False, f"brak wag {cls.MODEL} (uruchom tracking/fetch_weights.py)"
        return True, ""

    def _make(self):
        p = cv2.TrackerVit_Params()
        p.net = os.path.join(WEIGHTS_DIR, self.MODEL)
        return cv2.TrackerVit_create(p)


class DaSiamSOT(_CvSOT):
    info = SOTInfo("dasiamrpn", "siec syjamska RPN (ONNX)", True, "~200 MB wag, ciezki")
    _ctor = "TrackerDaSiamRPN"
    MODEL = "dasiamrpn_model.onnx"
    KERNEL_R1 = "dasiamrpn_kernel_r1.onnx"
    KERNEL_CLS1 = "dasiamrpn_kernel_cls1.onnx"

    @classmethod
    def available(cls):
        ok, msg = super().available()
        if not ok:
            return ok, msg
        for f in (cls.MODEL, cls.KERNEL_R1, cls.KERNEL_CLS1):
            if not os.path.exists(os.path.join(WEIGHTS_DIR, f)):
                return False, f"brak wag {f} (uruchom tracking/fetch_weights.py)"
        return True, ""

    def _make(self):
        p = cv2.TrackerDaSiamRPN_Params()
        p.model = os.path.join(WEIGHTS_DIR, self.MODEL)
        p.kernel_r1 = os.path.join(WEIGHTS_DIR, self.KERNEL_R1)
        p.kernel_cls1 = os.path.join(WEIGHTS_DIR, self.KERNEL_CLS1)
        return cv2.TrackerDaSiamRPN_create(p)


# ================================================= 5. "brak SOT" - sam Kalman
class NoneSOT(BaseSOT):
    """Placeholder: zadnego sledzenia wizyjnego miedzy detekcjami.
    Pozycja pochodzi wylacznie z predykcji Kalmana. Zerowy koszt - sluzy jako
    dolna granica w benchmarku."""

    info = SOTInfo("none", "tylko Kalman", False, "koszt zerowy, punkt odniesienia")

    def init(self, frame_bgr, bbox_xywh):
        self.box = tuple(float(v) for v in bbox_xywh)

    def update(self, frame_bgr):
        return False, self.box, 0.0


# ------------------------------------------------------------------ rejestr
REGISTRY: dict[str, type] = {
    "mosse": MosseSOT,
    "ncc": NccSOT,
    "medianflow": MedianFlowSOT,
    "mil": MilSOT,
    "kcf": KcfSOT,
    "csrt": CsrtSOT,
    "nanotrack": NanoSOT,
    "vittrack": VitSOT,
    "dasiamrpn": DaSiamSOT,
    "none": NoneSOT,
}


def create_backend(name: str, **kw) -> BaseSOT:
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(f"nieznany backend '{name}'. Dostepne: {sorted(REGISTRY)}")
    cls = REGISTRY[key]
    ok, msg = cls.available()
    if not ok:
        raise RuntimeError(f"backend '{name}' niedostepny: {msg}")
    return cls(**kw)


def available_backends() -> list[tuple[str, bool, str]]:
    out = []
    for name, cls in REGISTRY.items():
        ok, msg = cls.available()
        out.append((name, ok, msg or cls.info.note))
    return out


if __name__ == "__main__":
    print(f"{'backend':<12} {'rodzina':<32} {'status'}")
    print("-" * 78)
    for name, cls in REGISTRY.items():
        ok, msg = cls.available()
        status = "OK" if ok else f"NIE - {msg}"
        print(f"{name:<12} {cls.info.family:<32} {status}")
