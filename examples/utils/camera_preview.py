#!/usr/bin/env python
"""
| File: camera_preview.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Camera preview for Isaac Sim examples.

Isaac Sim's Python often has OpenCV without working highgui (GUI: NONE or GTK not
linked), even when opencv-python is installed. Use cv2 inside Isaac for image ops;
for live windows this module tries, in order:

  1. OpenCV highgui in-process (only if the build reports a GUI backend)
  2. OpenCV highgui in a system-Python subprocess (recommended on Linux)
  3. Matplotlib interactive window
  4. Omniverse ui.ImageWithProvider
"""

import importlib.util
import os
import re
import site
import struct
import subprocess
import sys

import numpy as np

try:
    import carb
except ImportError:
    carb = None

_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opencv_preview_worker.py")
_SYSTEM_PYTHON_GUI = None
_HEADER = struct.Struct("!III")


def _log_info(msg):
    if carb is not None:
        carb.log_info(msg)


def _log_warn(msg):
    if carb is not None:
        carb.log_warn(msg)


def _log_error(msg):
    if carb is not None:
        carb.log_error(msg)


def _isaac_kit_site_packages():
    base = os.environ.get("ISAACSIM_PATH", "/isaac-sim")
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return os.path.join(base, "kit", "python", "lib", py_ver, "site-packages")


def _purge_cv2_modules():
    for key in list(sys.modules.keys()):
        if key == "cv2" or key.startswith("cv2."):
            del sys.modules[key]


def _is_headless_cv2(cv2_module):
    return "headless" in (getattr(cv2_module, "__file__", None) or "").lower()


def cv2_build_has_highgui(cv2_module):
    """True if getBuildInformation reports a non-NONE GUI backend."""
    try:
        info = cv2_module.getBuildInformation()
        match = re.search(r"^GUI:\s*(\S+)", info, re.MULTILINE)
        if match:
            return match.group(1).upper() != "NONE"
    except Exception:
        pass
    return False


def _cv2_search_paths():
    paths = []
    kit_site = _isaac_kit_site_packages()
    if os.path.isdir(os.path.join(kit_site, "cv2")):
        paths.append(kit_site)

    user_site = site.getusersitepackages()
    if user_site and os.path.isdir(os.path.join(user_site, "cv2")):
        paths.append(user_site)

    return paths


def _load_cv2_from_site_packages(site_dir):
    cv2_dir = os.path.join(site_dir, "cv2")
    cv2_init = os.path.join(cv2_dir, "__init__.py")
    if not os.path.isfile(cv2_init):
        return None

    _purge_cv2_modules()
    spec = importlib.util.spec_from_file_location(
        "cv2", cv2_init, submodule_search_locations=[cv2_dir]
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules["cv2"] = module
    spec.loader.exec_module(module)
    return module


def import_cv2():
    """Load cv2 from kit/user site (for processing). Does not require highgui."""
    if "cv2" in sys.modules:
        return sys.modules["cv2"]

    for path in _cv2_search_paths():
        cv2 = _load_cv2_from_site_packages(path)
        if cv2 is not None:
            return cv2

    import cv2 as cv2_fallback

    return cv2_fallback


def _find_system_python_with_opencv_gui():
    """Find a system Python (not Isaac) that can open cv2 windows."""
    global _SYSTEM_PYTHON_GUI
    if _SYSTEM_PYTHON_GUI is not None:
        return _SYSTEM_PYTHON_GUI if _SYSTEM_PYTHON_GUI else None

    import shutil

    candidates = []
    for name in ("python3", "python"):
        path = shutil.which(name)
        if path:
            candidates.append(path)

    for path in ("/usr/bin/python3", "/usr/local/bin/python3"):
        if path not in candidates and os.path.isfile(path):
            candidates.append(path)

    test_code = (
        "import cv2; "
        "cv2.namedWindow('__pegasus_test__', cv2.WINDOW_NORMAL); "
        "cv2.destroyWindow('__pegasus_test__'); "
        "cv2.waitKey(1)"
    )

    for python_exe in candidates:
        if "isaac-sim" in python_exe.lower():
            continue
        try:
            result = subprocess.run(
                [python_exe, "-c", test_code],
                capture_output=True,
                timeout=15,
                env=os.environ.copy(),
            )
            if result.returncode == 0:
                _SYSTEM_PYTHON_GUI = python_exe
                _log_info(f"Camera preview will use system Python for OpenCV GUI: {python_exe}")
                return python_exe
        except Exception:
            continue

    _SYSTEM_PYTHON_GUI = ""
    return None


def _to_uint8_rgb(image):
    rgb = np.asarray(image[..., :3])
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)
    return rgb


def _rgb_to_bgr(rgb):
    cv2 = import_cv2()
    rgb = _to_uint8_rgb(rgb)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class SubprocessOpenCvCameraPreview:
    """
    OpenCV window in a separate system-Python process.
    Isaac Sim's bundled cv2 can stay headless; imshow runs where GTK works.
    """

    def __init__(self, title="Drone camera", width=640, height=480):
        python_exe = _find_system_python_with_opencv_gui()
        if not python_exe:
            raise RuntimeError(
                "No system Python with working OpenCV highgui found. Install on the host:\n"
                "  sudo apt install python3-opencv libgtk-3-0 libcanberra-gtk3-module\n"
                "  # or: python3 -m pip install opencv-python"
            )

        if not os.path.isfile(_WORKER_SCRIPT):
            raise RuntimeError(f"Missing worker script: {_WORKER_SCRIPT}")

        self._title = title
        self._proc = subprocess.Popen(
            [python_exe, _WORKER_SCRIPT, title, str(width), str(height)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._open = True
        _log_info(f"OpenCV preview subprocess started (pid {self._proc.pid}): {title}")

    def update(self, rgb):
        if not self._open or self._proc.stdin is None:
            return
        if rgb is None or len(rgb.shape) < 3:
            return

        bgr = _rgb_to_bgr(rgb)
        h, w = bgr.shape[:2]
        try:
            self._proc.stdin.write(_HEADER.pack(h, w, 3))
            self._proc.stdin.write(bgr.tobytes())
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._open = False
            err = self._proc.stderr.read().decode("utf-8", errors="replace") if self._proc.stderr else ""
            _log_error(f"OpenCV preview subprocess exited.\n{err}")

    def close(self):
        if not self._open:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._open = False


class OpenCvCameraPreview:
    """In-process cv2.imshow (only when the loaded cv2 build has highgui)."""

    def __init__(self, title="Drone camera", width=640, height=480):
        self._cv2 = import_cv2()
        if not cv2_build_has_highgui(self._cv2):
            gui_line = "unknown"
            try:
                match = re.search(r"^GUI:\s*(.+)$", self._cv2.getBuildInformation(), re.MULTILINE)
                if match:
                    gui_line = match.group(1).strip()
            except Exception:
                pass
            raise RuntimeError(
                f"OpenCV in Isaac Sim has no GUI backend (GUI: {gui_line}). "
                f"cv2 path: {self._cv2.__file__}"
            )

        self._title = title
        self._width = width
        self._height = height
        self._open = False
        self._window_ready = False
        self._pending_rgb = None
        self._init_error = None
        self._update_subscription = None
        _log_info(f"Using in-process OpenCV for preview: {self._cv2.__file__}")
        self._schedule_main_thread_window()

    def _schedule_main_thread_window(self):
        import omni.kit.app

        def _try_open_window(_event):
            if self._window_ready or self._init_error is not None:
                return
            try:
                self._cv2.namedWindow(self._title, self._cv2.WINDOW_NORMAL)
                self._cv2.resizeWindow(self._title, self._width, self._height)
                self._window_ready = True
                self._open = True
                _log_info(f"OpenCV camera preview window opened: {self._title}")
                if self._pending_rgb is not None:
                    pending = self._pending_rgb
                    self._pending_rgb = None
                    self.update(pending)
            except Exception as exc:
                self._init_error = exc
                _log_error(f"OpenCV namedWindow failed: {exc}")
            finally:
                if self._update_subscription is not None:
                    self._update_subscription.unsubscribe()
                    self._update_subscription = None

        stream = omni.kit.app.get_app().get_update_event_stream()
        self._update_subscription = stream.create_subscription_to_pop(
            _try_open_window,
            name="pegasus_opencv_camera_preview_init",
        )

    @property
    def init_error(self):
        return self._init_error

    def update(self, rgb):
        if self._init_error is not None:
            return
        if not self._window_ready:
            self._pending_rgb = rgb
            return
        if rgb is None or len(rgb.shape) < 3:
            return

        bgr = _rgb_to_bgr(rgb)
        self._cv2.imshow(self._title, bgr)
        self._cv2.waitKey(1)

    def close(self):
        if self._update_subscription is not None:
            self._update_subscription.unsubscribe()
            self._update_subscription = None
        if self._open:
            try:
                self._cv2.destroyWindow(self._title)
                self._cv2.waitKey(1)
            except Exception:
                pass
        self._open = False
        self._window_ready = False


class MatplotlibCameraPreview:
    """Interactive matplotlib window (TkAgg/QtAgg)."""

    def __init__(self, title="Drone camera", width=640, height=480):
        import matplotlib

        for backend in ("TkAgg", "Qt5Agg", "QtAgg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue

        # If we still ended up on a non-interactive backend (e.g. Agg), fail fast so the
        # caller can fall back to omni.ui instead of emitting warnings every frame.
        backend_name = str(matplotlib.get_backend()).lower()
        if "agg" in backend_name:
            raise RuntimeError(f"Matplotlib backend is non-interactive: {matplotlib.get_backend()}")

        import matplotlib.pyplot as plt

        self._plt = plt
        self._fig, self._ax = plt.subplots(num=title, figsize=(width / 100.0, height / 100.0))
        try:
            self._fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        self._im = self._ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))
        self._ax.axis("off")
        plt.ion()
        plt.show(block=False)
        _log_info(f"Camera preview using matplotlib ({matplotlib.get_backend()})")

    def update(self, rgb):
        if rgb is None or len(rgb.shape) < 3:
            return
        rgb = _to_uint8_rgb(rgb)
        self._im.set_data(rgb)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)

    def close(self):
        try:
            self._plt.close(self._fig)
        except Exception:
            pass


class OmniUiCameraPreview:
    """Omniverse UI window (always available inside Isaac Sim)."""

    def __init__(self, title="Drone camera", width=640, height=480):
        import omni.ui as ui

        self._width = width
        self._height = height
        self._provider = ui.ByteImageProvider()
        self._set_provider_image(np.zeros((height, width, 3), dtype=np.uint8))

        self._window = ui.Window(
            title,
            width=width + 16,
            height=height + 40,
            flags=ui.WINDOW_FLAGS_NO_SCROLLBAR,
        )

        with self._window.frame:
            ui.ImageWithProvider(self._provider, width=width, height=height)

        _log_info("Camera preview using Omniverse UI")

    def _set_provider_image(self, rgb):
        h, w = rgb.shape[:2]
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        rgba = np.ascontiguousarray(np.concatenate((rgb, alpha), axis=2))
        self._provider.set_data_array(rgba, [w, h])

    def update(self, rgb):
        if rgb is None or len(rgb.shape) < 3:
            return
        rgb = _to_uint8_rgb(rgb)
        h, w = rgb.shape[:2]
        if w != self._width or h != self._height:
            self._width = w
            self._height = h
        self._set_provider_image(rgb)

    def close(self):
        if self._window is not None:
            self._window.visible = False
            self._window = None


def create_camera_preview(
    title="Drone camera",
    width=640,
    height=480,
    prefer_opencv=True,
    skip_inprocess_opencv=False,
):
    """
    Create a live camera preview. Returns (preview, backend_name).

    prefer_opencv: try OpenCV-based backends first.
    skip_inprocess_opencv: skip in-process highgui (e.g. after async failure).
    """
    if prefer_opencv:
        if not skip_inprocess_opencv:
            try:
                return OpenCvCameraPreview(title, width, height), "opencv"
            except RuntimeError as exc:
                _log_warn(f"In-process OpenCV preview unavailable: {exc}")

        try:
            return SubprocessOpenCvCameraPreview(title, width, height), "opencv-subprocess"
        except RuntimeError as exc:
            _log_warn(f"Subprocess OpenCV preview unavailable: {exc}")

        try:
            return MatplotlibCameraPreview(title, width, height), "matplotlib"
        except Exception as exc:
            _log_warn(f"Matplotlib preview unavailable: {exc}")

    return OmniUiCameraPreview(title, width, height), "omni.ui"
