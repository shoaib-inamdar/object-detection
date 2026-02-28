"""
Voice Announcer — speaks detected object names aloud using pyttsx3.

Conditions for speaking:
    1. A NEW object label appears (not already being tracked)
    2. Confidence > VOICE_CONFIDENCE_THRESHOLD (default 60%)
    3. Object bounding box centre is within the centre region of the frame
    4. 3-second cooldown per object label (won't repeat the same label too soon)

Speech runs in a background thread so it never blocks the video loop.
"""

import threading
import time

import pyttsx3

import config


class VoiceAnnouncer:
    """Non-blocking text-to-speech announcer for detected objects."""

    def __init__(self):
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", config.VOICE_RATE)
        self._engine.setProperty("volume", config.VOICE_VOLUME)

        # {label: last_spoken_timestamp}
        self._cooldowns: dict[str, float] = {}
        # Labels that were present in the *previous* frame
        self._prev_labels: set[str] = set()

        self._lock = threading.Lock()
        self._speech_thread: threading.Thread | None = None

        print("[INFO] Voice announcer initialised.")

    # ── Public API ───────────────────────────────

    def update(self, detections: list[dict], frame_width: int, frame_height: int) -> list[str]:
        """
        Check detections and speak new, high-confidence, centre-frame objects.

        Args:
            detections:   list of detection dicts from ObjectDetector.detect()
            frame_width:  width of the current frame in pixels
            frame_height: height of the current frame in pixels

        Returns:
            list of labels that were spoken this frame (for testing / display)
        """
        if not config.VOICE_ENABLED:
            return []

        now = time.time()
        current_labels: set[str] = set()
        to_speak: list[str] = []

        for det in detections:
            label = det["label"]
            conf = det["confidence"]
            current_labels.add(label)

            # Condition 1: must be a NEW label (not in previous frame)
            if label in self._prev_labels:
                continue

            # Condition 2: confidence must exceed voice threshold
            if conf < config.VOICE_CONFIDENCE_THRESHOLD:
                continue

            # Condition 3: object centre must be in the centre region
            if not self._is_in_centre(det["box"], frame_width, frame_height):
                continue

            # Condition 4: cooldown — skip if spoken recently
            with self._lock:
                last = self._cooldowns.get(label, 0.0)
                if now - last < config.VOICE_COOLDOWN:
                    continue
                self._cooldowns[label] = now

            to_speak.append(label)

        # Update previous frame labels
        self._prev_labels = current_labels

        # Speak in background thread (non-blocking)
        if to_speak:
            self._speak_async(to_speak)

        return to_speak

    def shutdown(self):
        """Clean up the TTS engine."""
        try:
            self._engine.stop()
        except Exception:
            pass

    # ── Internals ────────────────────────────────

    @staticmethod
    def _is_in_centre(box, frame_w: int, frame_h: int) -> bool:
        """
        Check whether the centre of a bounding box falls within the
        centre region of the frame.

        The centre region is defined by VOICE_CENTER_FRACTION in config
        (default 0.5 means the middle 50% of the frame).
        """
        x, y, w, h = box
        cx = x + w / 2
        cy = y + h / 2

        frac = config.VOICE_CENTER_FRACTION
        margin_x = frame_w * (1 - frac) / 2
        margin_y = frame_h * (1 - frac) / 2

        return (margin_x <= cx <= frame_w - margin_x and
                margin_y <= cy <= frame_h - margin_y)

    def _speak_async(self, labels: list[str]):
        """Speak labels in a background thread so the video loop is not blocked."""
        # If previous speech is still running, skip to avoid queue build-up
        if self._speech_thread is not None and self._speech_thread.is_alive():
            return

        def _run():
            try:
                # Build a natural sentence
                text = ", ".join(labels)
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                # Silently ignore TTS errors to keep video running
                print(f"[WARN] TTS error: {e}")

        self._speech_thread = threading.Thread(target=_run, daemon=True)
        self._speech_thread.start()
