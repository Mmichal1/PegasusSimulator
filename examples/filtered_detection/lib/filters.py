"""Warstwa filtracji miedzy detekcja a wyjsciem dla ukladu naprowadzania.

Surowa detekcja to nie jest sygnal, ktory mozna podac do petli sterowania.
Trzy rodzaje szumu psuja namierzanie, kazdy wymaga innego mechanizmu:

  1. FALSZYWE ZABLOKOWANIE - detektor raz na jakis czas znajdzie ptaka albo
     chmure z pewnoscia 0.5. Jedna klatka wystarczy, zeby uklad zablokowal sie
     na smieciu. Lekarstwo: potwierdzenie M z N (`LockConfirmer`) - reguła
     znana z radarow, gdzie sciezke uznaje sie za realna dopiero po M trafieniu
     w oknie N skanow.

  2. PODMIANA CELU - po przeslonieciu detektor znajduje *jakis* obiekt w ROI
     i uklad przejmuje go jako "ten sam" cel. Dokladnie to BoT-SORT rozwiazywal
     przez ReID. Lekarstwo: lekki deskryptor wygladu (`AppearanceModel`)
     sprawdzany przy kazdym ponownym przejeciu.

  3. SKOK NA WYJSCIU - w chwili korekty detektorem estymata przeskakuje
     o kilkadziesiat do kilkuset pikseli. Dla regulatora to wymuszenie skokowe.
     Lekarstwo: ogranicznik szybkosci narastania (`OutputLimiter`) nalozony
     na WYJSCIE, nie na stan filtru - filtr dalej zbiega szybko, a punkt
     celowania przesuwa sie gladko, z flaga `settling` dla regulatora.

Czwarty element (`NISMonitor`) nie filtruje, tylko pilnuje, czy filtr Kalmana
jest w ogole spojny z danymi - bez tego strojenie R i Q jest zgadywaniem.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np


# ====================================================== 1. potwierdzenie locka
class LockConfirmer:
    """Regula M z N: sciezke uznajemy za realna po `hits` potwierdzeniach
    detektora w oknie `window` klatek. Do tego czasu tor jest 'wstepny' -
    jest sledzony wewnetrznie, ale NIE jest raportowany jako lock."""

    def __init__(self, hits: int = 3, window: int = 5, max_age: int = 12):
        self.hits = hits
        self.window = window
        self.max_age = max_age
        self.reset()

    def reset(self) -> None:
        self.buf: deque[bool] = deque(maxlen=self.window)
        self.age = 0

    def push(self, hit: bool) -> None:
        self.buf.append(bool(hit))
        self.age += 1

    @property
    def confirmed(self) -> bool:
        return sum(self.buf) >= self.hits

    @property
    def expired(self) -> bool:
        """Tor wstepny, ktory nie zebral potwierdzen - porzucamy."""
        return self.age >= self.max_age and not self.confirmed

    @property
    def progress(self) -> str:
        return f"{sum(self.buf)}/{self.hits}"


# ======================================================== 2. deskryptor wygladu
class AppearanceModel:
    """Kompaktowy opis wygladu celu: histogram HSV (odporny na zmiane jasnosci)
    plus histogram orientacji gradientu (lapie ksztalt/teksture).

    Koszt to ~0.2 ms na patch 32x32 - pomijalny wobec reszty pipeline'u,
    a chroni przed najgorszym trybem awarii: cichym przejsciem na inny obiekt.
    """

    def __init__(self, bins_hs: int = 8, bins_ori: int = 9, lr: float = 0.05,
                 patch: int = 32):
        self.bins_hs = bins_hs
        self.bins_ori = bins_ori
        self.lr = lr
        self.patch = patch
        self.desc: np.ndarray | None = None

    # ------------------------------------------------------------- deskryptor
    def _describe(self, frame_bgr: np.ndarray, bbox) -> np.ndarray | None:
        x, y, w, h = [int(round(v)) for v in bbox]
        H, W = frame_bgr.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + max(w, 2)), min(H, y + max(h, 2))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        roi = cv2.resize(frame_bgr[y0:y1, x0:x1], (self.patch, self.patch),
                         interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [self.bins_hs, self.bins_hs],
                            [0, 180, 0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-6)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = (np.arctan2(gy, gx) + np.pi) * (self.bins_ori / (2 * np.pi))
        ori = np.zeros(self.bins_ori, dtype=np.float32)
        idx = np.clip(ang.astype(int), 0, self.bins_ori - 1)
        np.add.at(ori, idx.ravel(), mag.ravel())
        ori = ori / (ori.sum() + 1e-6)

        return np.concatenate([hist, ori]).astype(np.float32)

    # -------------------------------------------------------------------- API
    def init(self, frame_bgr, bbox) -> None:
        d = self._describe(frame_bgr, bbox)
        if d is not None:
            self.desc = d

    def update(self, frame_bgr, bbox, lr: float | None = None) -> None:
        """Powolna adaptacja - cel zmienia orientacje i oswietlenie, ale
        zbyt szybka adaptacja zjadlaby caly sens kontroli."""
        d = self._describe(frame_bgr, bbox)
        if d is None:
            return
        if self.desc is None:
            self.desc = d
            return
        a = self.lr if lr is None else lr
        self.desc = (1 - a) * self.desc + a * d

    def similarity(self, frame_bgr, bbox) -> float:
        """Podobienstwo w [0, 1]: korelacja obu czesci deskryptora.
        Zwraca 1.0 gdy model nie jest jeszcze zainicjalizowany (brak wiedzy
        nie moze byc powodem do odrzucenia)."""
        if self.desc is None:
            return 1.0
        d = self._describe(frame_bgr, bbox)
        if d is None:
            return 0.0
        a, b = self.desc, d
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.clip(a @ b / (na * nb), 0.0, 1.0))


# ================================================ 3. ogranicznik skoku wyjscia
class OutputLimiter:
    """Ogranicznik nadmiarowego skoku punktu celowania.

    Prog jest wyliczany jako PRZEWIDZIANE przemieszczenie plus maly margines:

        dopuszczalne = vel_gain * |v| + base_px

    Pomiar na materiale z projektu pokazal, ze mediana stosunku skoku estymaty
    do |v| wynosi 1.00 - czyli typowy "skok" to po prostu realny ruch celu
    (mediana 5 px/klatke, p95 30 px/klatke). Prog proporcjonalny do |v|
    przepuszcza ten ruch bez tlumienia i reaguje tylko na nadwyzke ponad niego.

    Anomalia, ktora faktycznie psuje petle sterowania, to dociagniecie estymaty
    przez detektor po okresie zliczania drogi (przejscie `kalman -> detector`):
    mediana 78 px, maksimum 118 px przy |v| = 30. Filtr ma wtedy racje - duza
    kowariancja uzasadnia duze wzmocnienie Kalmana - ale regulator dostaje
    wymuszenie skokowe. Ogranicznik rozklada te korekte na kilka klatek
    i podnosi flage `settling`, zeby regulator mogl zmniejszyc wzmocnienie.

    Nadganianie jest progresywne (`ramp`), a nie skokowe: dopuszczalny krok
    rosnie z kazda klatka zalegania, wiec wyjscie dochodzi do estymaty gladko
    zamiast teleportowac sie po ustalonej liczbie klatek.

    Uwaga projektowa: po ograniczeniu punkt celowania przez kilka klatek
    ODSTAJE od najlepszej estymaty. To swiadomy kompromis - gladkosc kosztem
    chwilowego opoznienia. `TrackState.centroid_raw` niesie wersje bez
    ograniczenia, gdyby regulator wolal surowa estymate.
    """

    def __init__(self, base_px: float = 8.0, vel_gain: float = 1.0,
                 ramp: float = 0.6, max_lag: int = 12):
        self.base_px = base_px
        self.vel_gain = vel_gain
        self.ramp = ramp
        self.max_lag = max_lag
        self.reset()

    def reset(self) -> None:
        self.out: tuple[float, float] | None = None
        self.lag_frames = 0
        self.settling = False
        self.last_clip_px = 0.0
        self.n_clipped = 0

    def __call__(self, target, velocity=(0.0, 0.0)):
        if target is None:
            self.reset()
            return None
        if self.out is None:
            self.out = (float(target[0]), float(target[1]))
            return self.out
        dx = target[0] - self.out[0]
        dy = target[1] - self.out[1]
        d = float(np.hypot(dx, dy))
        allowed = self.vel_gain * float(np.hypot(*velocity)) + self.base_px
        allowed *= (1.0 + self.ramp * self.lag_frames)   # progresywne nadganianie
        if d <= allowed or self.lag_frames >= self.max_lag:
            self.out = (float(target[0]), float(target[1]))
            self.lag_frames = 0
            self.settling = False
            self.last_clip_px = 0.0
        else:
            k = allowed / d
            self.out = (self.out[0] + dx * k, self.out[1] + dy * k)
            self.lag_frames += 1
            self.settling = True
            self.last_clip_px = d - allowed
            self.n_clipped += 1
        return self.out


# ============================================== 4. kontrola spojnosci filtru
class NISMonitor:
    """Normalized Innovation Squared - test spojnosci filtru Kalmana.

    Dla pomiaru 2-wymiarowego (srodek celu) wartosc oczekiwana NIS wynosi 2.
    Srednia znaczaco powyzej 2 oznacza, ze filtr jest zbyt pewny siebie
    (R albo Q za male) i bedzie odrzucal poprawne pomiary; znaczaco ponizej -
    ze jest zbyt luzny i przepusci szum. Bez tej liczby strojenie filtru
    jest zgadywaniem.
    """

    def __init__(self, window: int = 60, dof: int = 2, min_samples: int = 15):
        self.window = window
        self.dof = dof
        # Ponizej tej liczby korekt detektora srednia NIS jest zdominowana przez
        # wariancje probkowania i potrafi wskazac "filtr zbyt luzny" na materiale,
        # na ktorym przy pelnej dlugosci wychodzi wartosc poprawna. Lepiej
        # powiedziec "za malo danych" niz podac mylacy werdykt.
        self.min_samples = min_samples
        self.buf: deque[float] = deque(maxlen=window)

    def push(self, nis: float) -> None:
        if np.isfinite(nis):
            self.buf.append(float(nis))

    @property
    def mean(self) -> float:
        return float(np.mean(self.buf)) if self.buf else float("nan")

    @property
    def ready(self) -> bool:
        return len(self.buf) >= self.min_samples

    @property
    def consistent(self) -> bool:
        """Zgrubny test: srednia NIS w przedziale [0.5, 2] * dof.
        Przy zbyt malej probie zwraca True - brak danych nie jest dowodem
        niespojnosci i nie powinien obnizac jakosci sygnalu."""
        if not self.ready:
            return True
        m = self.mean
        return bool(np.isfinite(m) and 0.5 * self.dof <= m <= 2.0 * self.dof)

    def verdict(self) -> str:
        m = self.mean
        if not np.isfinite(m):
            return "brak danych"
        if not self.ready:
            return f"NIS {m:.1f} - za malo danych ({len(self.buf)}/{self.min_samples} korekt)"
        if m > 2.0 * self.dof:
            return f"NIS {m:.1f} > {2.0*self.dof:.0f} - filtr zbyt pewny (zwieksz R lub Q)"
        if m < 0.5 * self.dof:
            return f"NIS {m:.1f} < {0.5*self.dof:.0f} - filtr zbyt luzny (zmniejsz R)"
        return f"NIS {m:.1f} - spojny"
