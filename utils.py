"""
Utility helpers: drawing overlays, FPS tracker, and HUD rendering.
"""

import time
import cv2
import config


# ──────────────────────────────────────────────
#  FPS Tracker
# ──────────────────────────────────────────────
class FPSTracker:
    """Smooth FPS counter using exponential moving average."""

    def __init__(self, avg_window: int = 30):
        self._prev_time = time.perf_counter()
        self._fps = 0.0
        self._alpha = 2 / (avg_window + 1)

    def update(self) -> float:
        now = time.perf_counter()
        dt = now - self._prev_time
        self._prev_time = now
        instant_fps = 1.0 / max(dt, 1e-6)
        self._fps = self._alpha * instant_fps + (1 - self._alpha) * self._fps
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# ──────────────────────────────────────────────
#  Drawing Functions
# ──────────────────────────────────────────────
def draw_detections(frame, detections):
    """
    Draw bounding boxes, labels, and confidence on the frame.

    Args:
        frame: BGR image (numpy array).
        detections: list of detection dicts from ObjectDetector.detect().
    """
    for det in detections:
        x, y, w, h = det["box"]
        color = det["color"]
        label = det["label"]
        conf = det["confidence"]

        # Bounding box (rounded corners effect via thickness)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label background
        text = ""
        if config.SHOW_CLASS_LABEL:
            text += label.upper()
        if config.SHOW_CONFIDENCE:
            text += f" {conf:.0%}"

        if text:
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            # Filled rectangle behind text
            cv2.rectangle(
                frame,
                (x, y - th - baseline - 6),
                (x + tw + 6, y),
                color,
                cv2.FILLED,
            )
            # Determine text color (white or black) for readability
            brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            txt_color = (0, 0, 0) if brightness > 160 else (255, 255, 255)
            cv2.putText(
                frame,
                text,
                (x + 3, y - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                txt_color,
                1,
                cv2.LINE_AA,
            )

    return frame


def draw_hud(frame, fps, detection_count):
    """
    Draw a heads-up display with FPS and object count.

    Args:
        frame: BGR image.
        fps: Current FPS value.
        detection_count: Number of detected objects.
    """
    h, w = frame.shape[:2]

    # Semi-transparent overlay bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if config.SHOW_FPS:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, fps_text, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )

    obj_text = f"Objects: {detection_count}"
    cv2.putText(
        frame, obj_text, (w - 200, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
    )

    # Instruction text at the bottom
    voice_state = "ON" if config.VOICE_ENABLED else "OFF"
    help_text = f"[q]uit [s]creenshot [p]ause [+/-]confidence [v]oice:{voice_state}"
    cv2.putText(
        frame, help_text, (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
    )

    return frame


def draw_info_panel(frame, detections):
    """
    Draw a small panel in the top-right corner listing detected object counts.

    Args:
        frame: BGR image (numpy array).
        detections: list of detection dicts.
    """
    if not detections:
        return frame

    # Count objects per class
    counts = {}
    for det in detections:
        label = det["label"]
        counts[label] = counts.get(label, 0) + 1

    h, w = frame.shape[:2]
    panel_h = 30 + len(counts) * 22
    panel_w = 200

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 45), (w, 45 + panel_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame, "Detected:", (w - panel_w + 10, 65),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
    )
    for i, (label, count) in enumerate(sorted(counts.items())):
        y_pos = 88 + i * 22
        cv2.putText(
            frame, f"  {label}: {count}", (w - panel_w + 10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return frame
