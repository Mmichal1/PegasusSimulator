#!/usr/bin/env python
"""
| File: vertical_circular_controller.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Horizontal circle at a fixed altitude (for targets above a pursuer).
"""

from circular_controller import CircularController


class VerticalCircularController(CircularController):
    """
    Alias for a horizontal circle at constant height.

    Use center=(pursuer_x, pursuer_y) and height=target_altitude so the target orbits
    directly above the pursuer's horizontal position.
    """

    pass
