#!/usr/bin/env python
"""
| File: red_ball_tracker.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: HSV-based red blob detection for visual guidance.
"""

import numpy as np

from camera_preview import import_cv2


def _prepare_rgb(rgb_image):
    rgb = np.asarray(rgb_image)
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)
    return rgb


def _red_mask(bgr, cv2):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_red_a = np.array([0, 120, 80], dtype=np.uint8)
    upper_red_a = np.array([12, 255, 255], dtype=np.uint8)
    lower_red_b = np.array([165, 120, 80], dtype=np.uint8)
    upper_red_b = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_red_a, upper_red_a) | cv2.inRange(hsv, lower_red_b, upper_red_b)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=3)
    return mask


def detect_red_balls(rgb_image, min_area=120):
    """Find all red blobs in an RGB uint8 image."""
    cv2 = import_cv2()
    rgb = _prepare_rgb(rgb_image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return {"detected": False, "blobs": [], "mask": None}

    bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
    mask = _red_mask(bgr, cv2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for contour in contours or []:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            continue
        u = int(moments["m10"] / moments["m00"])
        v = int(moments["m01"] / moments["m00"])
        blobs.append({"centroid": (u, v), "area": area})

    blobs.sort(key=lambda b: b["area"], reverse=True)
    return {"detected": len(blobs) > 0, "blobs": blobs, "mask": mask}


def detect_red_ball(rgb_image, min_area=120):
    """Largest red blob (single-target helper)."""
    result = detect_red_balls(rgb_image, min_area=min_area)
    if not result["blobs"]:
        return {"detected": False, "centroid": None, "area": 0.0, "mask": result["mask"]}
    best = result["blobs"][0]
    return {
        "detected": True,
        "centroid": best["centroid"],
        "area": best["area"],
        "mask": result["mask"],
    }


def select_blob_for_pursuit(blobs, image_width, image_height, forward_u_fraction=0.25):
    """
    Pick the blob to chase for endless visual pursuit.

    Prefers red objects near the image centre and in the forward part of the view.
    As the drone passes one ball, another naturally moves toward the centre.
    """
    if not blobs:
        return None

    cx = 0.5 * image_width
    cy = 0.5 * image_height
    u_min = forward_u_fraction * image_width

    def pursuit_cost(blob):
        u, v = blob["centroid"]
        center_dist = (u - cx) ** 2 + (v - cy) ** 2
        if u < u_min:
            center_dist += 1e6
        return center_dist - 0.02 * blob["area"]

    return min(blobs, key=pursuit_cost)


def draw_detection(rgb_image, detection):
    """Draw crosshair, all blobs, and highlight the actively pursued target."""
    cv2 = import_cv2()
    rgb = _prepare_rgb(rgb_image[..., :3]).copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.drawMarker(bgr, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)

    blobs = detection.get("blobs", [])
    active_centroid = detection.get("centroid")

    for blob in blobs:
        u, v = blob["centroid"]
        is_active = active_centroid is not None and blob["centroid"] == active_centroid
        color = (0, 255, 0) if is_active else (0, 180, 255)
        thickness = 3 if is_active else 1
        cv2.circle(bgr, (u, v), 10 if is_active else 6, color, thickness)

    if active_centroid is not None:
        u, v = active_centroid
        cv2.line(bgr, (cx, cy), (u, v), (255, 200, 0), 2)
        cv2.putText(
            bgr,
            "target",
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        status = detection.get("status", "searching")
        cv2.putText(
            bgr,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if detection.get("startup_hold"):
        cv2.putText(
            bgr,
            "STANDBY",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
