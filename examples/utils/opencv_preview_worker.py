#!/usr/bin/env python3
"""
Standalone worker: reads BGR frames from stdin and shows them with cv2.imshow.
Spawned by camera_preview.SubprocessOpenCvCameraPreview using system Python
(where OpenCV highgui is usually linked against GTK correctly).
"""

import struct
import sys

import cv2
import numpy as np


def main():
    title = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, width, height)

    header = struct.Struct("!III")
    stdin = sys.stdin.buffer

    while True:
        meta = stdin.read(header.size)
        if len(meta) < header.size:
            break

        h, w, channels = header.unpack(meta)
        if channels != 3 or h <= 0 or w <= 0:
            break

        nbytes = h * w * channels
        raw = stdin.read(nbytes)
        if len(raw) < nbytes:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, channels)
        cv2.imshow(title, frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
