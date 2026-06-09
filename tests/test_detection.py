import os
import sys
import time

import cv2
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from detector import ObjectDetector
from utils import FPSTracker, draw_detections, draw_hud, draw_info_panel
from voice import VoiceAnnouncer

TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "tests", "test_images")


@pytest.fixture(scope="module")
def detector():
    return ObjectDetector()


@pytest.fixture
def dog_image():
    path = os.path.join(TEST_IMAGES_DIR, "dog.jpg")
    img = cv2.imread(path)
    assert img is not None, f"Test image not found: {path}"
    return img


@pytest.fixture
def horses_image():
    path = os.path.join(TEST_IMAGES_DIR, "horses.jpg")
    img = cv2.imread(path)
    assert img is not None, f"Test image not found: {path}"
    return img


@pytest.fixture
def person_image():
    path = os.path.join(TEST_IMAGES_DIR, "person.jpg")
    img = cv2.imread(path)
    assert img is not None, f"Test image not found: {path}"
    return img


@pytest.fixture
def kite_image():
    path = os.path.join(TEST_IMAGES_DIR, "kite.jpg")
    img = cv2.imread(path)
    assert img is not None, f"Test image not found: {path}"
    return img


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def noise_frame():
    np.random.seed(123)
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


class TestConfig:
    def test_confidence_threshold_range(self):
        assert 0.0 < config.CONFIDENCE_THRESHOLD < 1.0

    def test_nms_threshold_range(self):
        assert 0.0 < config.NMS_THRESHOLD < 1.0

    def test_input_size_valid(self):
        w, h = config.INPUT_SIZE
        assert w > 0 and h > 0
        assert w % 32 == 0 and h % 32 == 0

    def test_paths_defined(self):
        assert config.YOLO_WEIGHTS is not None
        assert config.YOLO_CONFIG is not None
        assert config.COCO_NAMES is not None


class TestModelFiles:
    def test_weights_exist(self):
        assert os.path.isfile(config.YOLO_WEIGHTS)

    def test_config_exist(self):
        assert os.path.isfile(config.YOLO_CONFIG)

    def test_names_exist(self):
        assert os.path.isfile(config.COCO_NAMES)

    def test_weights_not_empty(self):
        size = os.path.getsize(config.YOLO_WEIGHTS)
        assert size > 1_000_000

    def test_coco_names_has_80_classes(self):
        with open(config.COCO_NAMES, "r") as f:
            classes = [line.strip() for line in f if line.strip()]
        assert len(classes) == 80

    def test_coco_names_starts_with_person(self):
        with open(config.COCO_NAMES, "r") as f:
            first = f.readline().strip()
        assert first == "person"


class TestDetectorInit:
    def test_detector_creates(self, detector):
        assert detector is not None

    def test_class_names_loaded(self, detector):
        assert len(detector.class_names) == 80

    def test_network_loaded(self, detector):
        assert detector.net is not None

    def test_output_layers_exist(self, detector):
        assert len(detector.output_layers) >= 1

    def test_colors_generated(self, detector):
        assert detector.colors.shape == (80, 3)


class TestMultiClassDetection:
    def test_dog_image_detects_dog(self, detector, dog_image):
        dets = detector.detect(dog_image)
        labels = {d["label"] for d in dets}
        assert "dog" in labels

    def test_dog_image_detects_multiple_classes(self, detector, dog_image):
        dets = detector.detect(dog_image)
        labels = {d["label"] for d in dets}
        assert len(labels) >= 2

    def test_dog_image_detects_bicycle_or_truck(self, detector, dog_image):
        dets = detector.detect(dog_image)
        labels = {d["label"] for d in dets}
        has_vehicle = "bicycle" in labels or "truck" in labels or "car" in labels
        assert has_vehicle

    def test_horses_image_detects_horses(self, detector, horses_image):
        dets = detector.detect(horses_image)
        labels = {d["label"] for d in dets}
        has_animal = "horse" in labels or "cow" in labels
        assert has_animal

    def test_person_image_detects_person_and_dog(self, detector, person_image):
        dets = detector.detect(person_image)
        labels = {d["label"] for d in dets}
        assert "person" in labels
        assert "dog" in labels

    def test_kite_image_detects_kite(self, detector, kite_image):
        dets = detector.detect(kite_image)
        labels = {d["label"] for d in dets}
        assert "kite" in labels

    def test_kite_image_detects_person(self, detector, kite_image):
        dets = detector.detect(kite_image)
        labels = {d["label"] for d in dets}
        assert "person" in labels

    def test_total_unique_classes_across_images(self, detector, dog_image, horses_image, person_image, kite_image):
        all_labels = set()
        for img in [dog_image, horses_image, person_image, kite_image]:
            dets = detector.detect(img)
            all_labels.update(d["label"] for d in dets)
        assert len(all_labels) >= 5


class TestDetectionFormat:
    def test_detection_has_required_keys(self, detector, dog_image):
        dets = detector.detect(dog_image)
        assert len(dets) > 0
        required_keys = {"class_id", "label", "confidence", "box", "color"}
        for det in dets:
            assert required_keys.issubset(det.keys())

    def test_confidence_in_valid_range(self, detector, dog_image):
        dets = detector.detect(dog_image)
        for det in dets:
            assert 0.0 < det["confidence"] <= 1.0

    def test_box_has_four_values(self, detector, dog_image):
        dets = detector.detect(dog_image)
        for det in dets:
            assert len(det["box"]) == 4

    def test_class_id_valid(self, detector, dog_image):
        dets = detector.detect(dog_image)
        for det in dets:
            assert 0 <= det["class_id"] < 80

    def test_color_is_bgr_tuple(self, detector, dog_image):
        dets = detector.detect(dog_image)
        for det in dets:
            assert len(det["color"]) == 3
            for c in det["color"]:
                assert 0 <= c <= 255


class TestConfidenceThreshold:
    def test_higher_threshold_fewer_detections(self, detector, dog_image):
        original = config.CONFIDENCE_THRESHOLD
        try:
            config.CONFIDENCE_THRESHOLD = 0.25
            dets_low = detector.detect(dog_image)
            config.CONFIDENCE_THRESHOLD = 0.70
            dets_high = detector.detect(dog_image)
            assert len(dets_high) <= len(dets_low)
        finally:
            config.CONFIDENCE_THRESHOLD = original

    def test_very_high_threshold_few_detections(self, detector, dog_image):
        original = config.CONFIDENCE_THRESHOLD
        try:
            config.CONFIDENCE_THRESHOLD = 0.95
            dets = detector.detect(dog_image)
            assert len(dets) <= 2
        finally:
            config.CONFIDENCE_THRESHOLD = original

    def test_low_threshold_more_classes(self, detector, dog_image):
        original = config.CONFIDENCE_THRESHOLD
        try:
            config.CONFIDENCE_THRESHOLD = 0.15
            labels_low = {d["label"] for d in detector.detect(dog_image)}
            config.CONFIDENCE_THRESHOLD = 0.60
            labels_high = {d["label"] for d in detector.detect(dog_image)}
            assert len(labels_low) >= len(labels_high)
        finally:
            config.CONFIDENCE_THRESHOLD = original


class TestEdgeCases:
    def test_blank_frame_no_crash(self, detector, blank_frame):
        dets = detector.detect(blank_frame)
        assert isinstance(dets, list)

    def test_noise_frame_no_crash(self, detector, noise_frame):
        dets = detector.detect(noise_frame)
        assert isinstance(dets, list)

    def test_tiny_frame(self, detector):
        tiny = np.zeros((32, 32, 3), dtype=np.uint8)
        dets = detector.detect(tiny)
        assert isinstance(dets, list)

    def test_large_frame(self, detector):
        large = np.zeros((1080, 1920, 3), dtype=np.uint8)
        dets = detector.detect(large)
        assert isinstance(dets, list)


class TestFPSTracker:
    def test_initial_fps_is_zero_or_positive(self):
        tracker = FPSTracker()
        assert tracker.fps >= 0.0

    def test_fps_updates(self):
        tracker = FPSTracker(avg_window=5)
        for _ in range(10):
            tracker.update()
            time.sleep(0.01)
        assert tracker.fps > 0.0

    def test_fps_reasonable_range(self):
        tracker = FPSTracker(avg_window=5)
        for _ in range(50):
            tracker.update()
            time.sleep(0.05)
        assert 5.0 < tracker.fps < 100.0


class TestDrawing:
    def _make_sample_detections(self):
        return [
            {"class_id": 0, "label": "person", "confidence": 0.95, "box": [100, 50, 200, 300], "color": (0, 255, 0)},
            {"class_id": 16, "label": "dog", "confidence": 0.87, "box": [300, 200, 150, 120], "color": (255, 0, 0)},
        ]

    def test_draw_detections_returns_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = self._make_sample_detections()
        result = draw_detections(frame, dets)
        assert result is not None
        assert result.shape == (480, 640, 3)

    def test_draw_detections_modifies_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_sum = frame.sum()
        dets = self._make_sample_detections()
        result = draw_detections(frame, dets)
        assert result.sum() > original_sum, "Drawing should add pixels to the frame"

    def test_draw_detections_empty_list(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_detections(frame, [])
        assert result is not None
        assert result.shape == (480, 640, 3)

    def test_draw_hud_returns_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_hud(frame, 30.0, 5)
        assert result is not None
        assert result.shape == (480, 640, 3)

    def test_draw_hud_adds_content(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_sum = frame.sum()
        result = draw_hud(frame, 30.0, 5)
        assert result.sum() > original_sum

    def test_draw_info_panel_with_detections(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = self._make_sample_detections()
        result = draw_info_panel(frame, dets)
        assert result is not None

    def test_draw_info_panel_empty_detections(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        result = draw_info_panel(frame, [])
        # Empty detections should return frame unchanged
        np.testing.assert_array_equal(result, original)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  10. Integration Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIntegration:
    def test_full_pipeline(self, detector, dog_image):
        """End-to-end: detect + draw + HUD on a real image."""
        # Detect
        dets = detector.detect(dog_image)
        assert len(dets) > 0

        # Draw
        frame = draw_detections(dog_image.copy(), dets)
        frame = draw_hud(frame, 25.0, len(dets))
        frame = draw_info_panel(frame, dets)

        # Output should be valid image
        assert frame is not None
        assert frame.shape == dog_image.shape
        assert frame.dtype == np.uint8

    def test_consecutive_detections_consistent(self, detector, person_image):
        """Running detection twice on same image should give same results."""
        dets1 = detector.detect(person_image)
        dets2 = detector.detect(person_image)

        labels1 = sorted(d["label"] for d in dets1)
        labels2 = sorted(d["label"] for d in dets2)
        assert labels1 == labels2, \
            f"Consecutive detections should be consistent: {labels1} vs {labels2}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  11. Voice Announcer Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVoiceAnnouncer:
    """Test the voice announcement logic (conditions, cooldowns, centre check)."""

    @pytest.fixture(autouse=True)
    def _save_config(self):
        """Save and restore voice config around each test."""
        orig_enabled = config.VOICE_ENABLED
        orig_threshold = config.VOICE_CONFIDENCE_THRESHOLD
        orig_cooldown = config.VOICE_COOLDOWN
        orig_fraction = config.VOICE_CENTER_FRACTION
        yield
        config.VOICE_ENABLED = orig_enabled
        config.VOICE_CONFIDENCE_THRESHOLD = orig_threshold
        config.VOICE_COOLDOWN = orig_cooldown
        config.VOICE_CENTER_FRACTION = orig_fraction

    @pytest.fixture
    def announcer(self):
        a = VoiceAnnouncer()
        yield a
        a.shutdown()

    def _make_detection(self, label, confidence, cx, cy, w=50, h=50):
        """Create a fake detection dict with box centred at (cx, cy)."""
        x = cx - w // 2
        y = cy - h // 2
        return {
            "class_id": 0,
            "label": label,
            "confidence": confidence,
            "box": [x, y, w, h],
            "color": (0, 255, 0),
        }

    def test_new_high_confidence_centre_object_is_spoken(self, announcer):
        """A new object with >60% confidence in the centre should be spoken."""
        config.VOICE_ENABLED = True
        config.VOICE_CONFIDENCE_THRESHOLD = 0.60
        config.VOICE_CENTER_FRACTION = 0.5
        config.VOICE_COOLDOWN = 3.0

        det = self._make_detection("dog", 0.85, cx=320, cy=240)
        spoken = announcer.update([det], frame_width=640, frame_height=480)
        assert "dog" in spoken

    def test_low_confidence_not_spoken(self, announcer):
        """Objects below 60% confidence should NOT be spoken."""
        config.VOICE_ENABLED = True
        config.VOICE_CONFIDENCE_THRESHOLD = 0.60

        det = self._make_detection("cat", 0.40, cx=320, cy=240)
        spoken = announcer.update([det], frame_width=640, frame_height=480)
        assert "cat" not in spoken

    def test_object_outside_centre_not_spoken(self, announcer):
        """Objects at the edge of the frame should NOT be spoken."""
        config.VOICE_ENABLED = True
        config.VOICE_CONFIDENCE_THRESHOLD = 0.60
        config.VOICE_CENTER_FRACTION = 0.5

        # Place object in the top-left corner (far from centre)
        det = self._make_detection("car", 0.90, cx=10, cy=10)
        spoken = announcer.update([det], frame_width=640, frame_height=480)
        assert "car" not in spoken

    def test_repeated_object_not_spoken_again(self, announcer):
        """Same object in consecutive frames should NOT be spoken again."""
        config.VOICE_ENABLED = True
        config.VOICE_CONFIDENCE_THRESHOLD = 0.60

        det = self._make_detection("person", 0.95, cx=320, cy=240)

        # Frame 1 — should speak
        spoken1 = announcer.update([det], 640, 480)
        assert "person" in spoken1

        # Frame 2 — same label still present → NOT new → should NOT speak
        spoken2 = announcer.update([det], 640, 480)
        assert "person" not in spoken2

    def test_reappearing_object_after_absence_respects_cooldown(self, announcer):
        """Object that disappears and reappears within cooldown should NOT speak."""
        config.VOICE_ENABLED = True
        config.VOICE_CONFIDENCE_THRESHOLD = 0.60
        config.VOICE_COOLDOWN = 3.0

        det = self._make_detection("dog", 0.85, cx=320, cy=240)

        # Frame 1 — speak
        spoken1 = announcer.update([det], 640, 480)
        assert "dog" in spoken1

        # Frame 2 — object gone
        announcer.update([], 640, 480)

        # Frame 3 — object back, but within cooldown
        spoken3 = announcer.update([det], 640, 480)
        assert "dog" not in spoken3

    def test_reappearing_object_after_cooldown_speaks(self, announcer):
        """Object that reappears AFTER cooldown period SHOULD be spoken."""
        config.VOICE_ENABLED = True
        config.VOICE_CONFIDENCE_THRESHOLD = 0.60
        config.VOICE_COOLDOWN = 0.1  # Very short for testing

        det = self._make_detection("cat", 0.85, cx=320, cy=240)

        # Frame 1 — speak
        spoken1 = announcer.update([det], 640, 480)
        assert "cat" in spoken1

        # Object disappears
        announcer.update([], 640, 480)

        # Wait for cooldown to expire
        time.sleep(0.15)

        # Object comes back — should speak again
        spoken3 = announcer.update([det], 640, 480)
        assert "cat" in spoken3

    def test_voice_disabled_nothing_spoken(self, announcer):
        """When VOICE_ENABLED is False, nothing should be spoken."""
        config.VOICE_ENABLED = False

        det = self._make_detection("person", 0.99, cx=320, cy=240)
        spoken = announcer.update([det], 640, 480)
        assert spoken == []

    def test_multiple_new_objects_spoken(self, announcer):
        """Multiple new objects in one frame should all be spoken."""
        config.VOICE_ENABLED = True
        config.VOICE_CONFIDENCE_THRESHOLD = 0.60

        dets = [
            self._make_detection("dog", 0.90, cx=320, cy=240),
            self._make_detection("cat", 0.75, cx=300, cy=260),
        ]
        spoken = announcer.update(dets, 640, 480)
        assert "dog" in spoken
        assert "cat" in spoken

    def test_centre_boundary(self, announcer):
        """Object exactly on the centre region boundary should be included."""
        config.VOICE_ENABLED = True
        config.VOICE_CONFIDENCE_THRESHOLD = 0.60
        config.VOICE_CENTER_FRACTION = 0.5

        # For 640x480 with fraction 0.5:
        # margin_x = 640 * 0.25 = 160, margin_y = 480 * 0.25 = 120
        # Centre region: x in [160, 480], y in [120, 360]
        det = self._make_detection("bottle", 0.80, cx=160, cy=120)
        spoken = announcer.update([det], 640, 480)
        assert "bottle" in spoken

    def test_is_in_centre_static_method(self):
        """Directly test the static _is_in_centre method."""
        config.VOICE_CENTER_FRACTION = 0.5

        # Centre of 640x480 → definitely in centre
        assert VoiceAnnouncer._is_in_centre([295, 215, 50, 50], 640, 480) is True

        # Top-left corner → NOT in centre
        assert VoiceAnnouncer._is_in_centre([0, 0, 20, 20], 640, 480) is False

        # Bottom-right corner → NOT in centre
        assert VoiceAnnouncer._is_in_centre([620, 460, 20, 20], 640, 480) is False
