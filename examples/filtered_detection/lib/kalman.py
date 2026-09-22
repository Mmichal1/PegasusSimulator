"""Filtr Kalmana dla celu w plaszczyznie obrazu.

Stan: [cx, cy, w, h, vx, vy]  (piksele / piksele-na-klatke)
Pomiar: [cx, cy, w, h]

Model ruchu: staloprędkościowy (CV) dla srodka + bladzenie losowe dla rozmiaru.
Filtr pelni trzy role w pipeline:
  1. wygladza centroid podawany do kontrolera naprowadzania,
  2. przewiduje pozycje w klatkach bez detekcji (coasting),
  3. zawęża okno ROI dla detektora -> glowna oszczednosc obliczeniowa.
"""
from __future__ import annotations

import numpy as np


class BoxKalman:
    def __init__(
        self,
        dt: float = 1.0,
        q_pos: float = 1.0,
        q_vel: float = 12.0,
        q_size: float = 6.0,
        r_pos: float = 4.0,
        r_size: float = 12.0,
        r_pos_frac: float = 0.02,
        r_size_frac: float = 0.0,
    ):
        self.dt = float(dt)
        self.q_pos, self.q_vel, self.q_size = q_pos, q_vel, q_size
        self.r_pos, self.r_size = r_pos, r_size
        # Szum pomiaru skaluje sie z rozmiarem celu. Srodek bboxa o boku 500 px
        # jest z natury gorzej okreslony niz srodek bboxa o boku 20 px - detektor
        # przesuwa krawedzie o kilka procent rozmiaru z klatki na klatke.
        # Staly r_pos zakladal niepewnosc ~2 px niezaleznie od skali celu, przez
        # co filtr byl zbyt pewny siebie (zmierzone NIS ~12 przy oczekiwanym 2)
        # i jego bramka odrzucala poprawne detekcje.
        # Dla POZYCJI skalowanie rozmiarem celu bylo istotna poprawka.
        # Dla ROZMIARU okazalo sie szkodliwe: zbyt wygladzony bbox powoduje
        # reinicjalizacje trackera z blednego pudelka i pogarsza rowniez blad
        # srodka (zmierzone: 6.9 px przy r_size_frac=0 wobec 14.5 px przy 0.05).
        # Stad domyslne zero - rozmiar zostaje przy stalym szumie pomiaru.
        self.r_pos_frac = r_pos_frac
        self.r_size_frac = r_size_frac
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1e3
        self.initialised = False

    # ---------------------------------------------------------------- macierze
    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 4] = dt
        F[1, 5] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        Q = np.zeros((6, 6))
        # bialy szum przyspieszenia dla (cx, vx) i (cy, vy)
        for p, v in ((0, 4), (1, 5)):
            Q[p, p] = self.q_pos * dt ** 3 / 3.0
            Q[p, v] = Q[v, p] = self.q_pos * dt ** 2 / 2.0
            Q[v, v] = self.q_vel * dt
        Q[2, 2] = Q[3, 3] = self.q_size * dt
        return Q

    @staticmethod
    def _H() -> np.ndarray:
        H = np.zeros((4, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        return H

    def _R(self, scale: float) -> np.ndarray:
        s = max(scale, 1e-3)
        side = max(float(self.x[2]), float(self.x[3]), 1.0) if self.initialised else 1.0
        rp = self.r_pos + (self.r_pos_frac * side) ** 2
        rs = self.r_size + (self.r_size_frac * side) ** 2
        return np.diag([rp * s, rp * s, rs * s, rs * s])

    # ------------------------------------------------------------------- API
    def reset(self, bbox_xywh, vel=(0.0, 0.0)) -> None:
        x, y, w, h = bbox_xywh
        self.x = np.array([x + w / 2.0, y + h / 2.0, w, h, vel[0], vel[1]], dtype=np.float64)
        self.P = np.diag([16.0, 16.0, 64.0, 64.0, 100.0, 100.0])
        self.initialised = True

    def predict(self, dt: float | None = None) -> np.ndarray:
        if not self.initialised:
            return self.x.copy()
        dt = self.dt if dt is None else dt
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        self.x[2] = max(self.x[2], 2.0)
        self.x[3] = max(self.x[3], 2.0)
        return self.x.copy()

    def update(self, bbox_xywh, meas_noise_scale: float = 1.0) -> float:
        """Korekta pomiarem. `meas_noise_scale` < 1 = ufamy bardziej (detektor),
        > 1 = ufamy mniej (tani tracker korelacyjny).

        Zwraca NIS (normalized innovation squared) dla skladowej pozycyjnej -
        miare zgodnosci pomiaru z predykcja, uzywana do kontroli spojnosci
        filtru. Wartosc oczekiwana dla 2 stopni swobody wynosi 2."""
        x, y, w, h = bbox_xywh
        z = np.array([x + w / 2.0, y + h / 2.0, w, h], dtype=np.float64)
        H = self._H()
        R = self._R(meas_noise_scale)
        y_res = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        nis = float(y_res[:2] @ np.linalg.inv(S[:2, :2]) @ y_res[:2])
        self.x = self.x + K @ y_res
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.x[2] = max(self.x[2], 2.0)
        self.x[3] = max(self.x[3], 2.0)
        return nis

    def gating_distance(self, bbox_xywh, meas_noise_scale: float = 1.0) -> float:
        """Odleglosc Mahalanobisa pomiaru od predykcji (tylko srodek)."""
        if not self.initialised:
            return 0.0
        x, y, w, h = bbox_xywh
        z = np.array([x + w / 2.0, y + h / 2.0])
        H = self._H()[:2]
        S = H @ self.P @ H.T + self._R(meas_noise_scale)[:2, :2]
        d = z - H @ self.x
        return float(np.sqrt(d @ np.linalg.inv(S) @ d))

    # ------------------------------------------------------------- odczyty
    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        cx, cy, w, h = self.x[:4]
        return (cx - w / 2.0, cy - h / 2.0, w, h)

    @property
    def center(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self.x[4]), float(self.x[5]))

    @property
    def pos_std(self) -> float:
        return float(np.sqrt(max(self.P[0, 0], 0.0) + max(self.P[1, 1], 0.0)))
