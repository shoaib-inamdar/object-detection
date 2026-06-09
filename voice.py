import threading
import time
import pyttsx3
import config


class VoiceAnnouncer:
    def __init__(self):
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", config.VOICE_RATE)
        self._engine.setProperty("volume", config.VOICE_VOLUME)
        self._cooldowns: dict[str, float] = {}
        self._prev_labels: set[str] = set()
        self._lock = threading.Lock()
        self._speech_thread: threading.Thread | None = None
        print("[INFO] Voice announcer initialised.")

    def update(self, detections: list[dict], frame_width: int, frame_height: int) -> list[str]:
        if not config.VOICE_ENABLED:
            return []
        now = time.time()
        current_labels: set[str] = set()
        to_speak: list[str] = []
        for det in detections:
            label = det["label"]
            conf = det["confidence"]
            current_labels.add(label)
            if conf < config.VOICE_CONFIDENCE_THRESHOLD:
                continue
            if not self._is_in_centre(det["box"], frame_width, frame_height):
                continue
            with self._lock:
                last = self._cooldowns.get(label, 0.0)
                if now - last < config.VOICE_COOLDOWN:
                    continue
                self._cooldowns[label] = now
            to_speak.append(label)
        self._prev_labels = current_labels
        if to_speak:
            self._speak_async(to_speak)
        return to_speak

    def shutdown(self):
        try:
            self._engine.stop()
        except Exception:
            pass

    @staticmethod
    def _is_in_centre(box, frame_w: int, frame_h: int) -> bool:
        x, y, w, h = box
        cx = x + w / 2
        cy = y + h / 2
        frac = config.VOICE_CENTER_FRACTION
        margin_x = frame_w * (1 - frac) / 2
        margin_y = frame_h * (1 - frac) / 2
        return (margin_x <= cx <= frame_w - margin_x and margin_y <= cy <= frame_h - margin_y)

    def _speak_async(self, labels: list[str]):
        if self._speech_thread is not None and self._speech_thread.is_alive():
            return
        def _run():
            try:
                text = ", ".join(labels)
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                print(f"[WARN] TTS error: {e}")
        self._speech_thread = threading.Thread(target=_run, daemon=True)
        self._speech_thread.start()
