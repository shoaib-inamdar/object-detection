"""
YOLOv4 Object Detector using OpenCV DNN module.
Handles model loading, inference, and post-processing.
"""

import cv2
import numpy as np
import config


class ObjectDetector:
    """Real-time object detector powered by YOLOv4-tiny and OpenCV DNN."""

    def __init__(self):
        self._load_class_names()
        self._load_network()
        self._generate_colors()

    # ── Model Loading ────────────────────────────
    def _load_class_names(self):
        """Load COCO class names from file."""
        with open(config.COCO_NAMES, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"[INFO] Loaded {len(self.class_names)} class labels.")

    def _load_network(self):
        """Load YOLOv4-tiny network with OpenCV DNN."""
        print("[INFO] Loading YOLO network...")
        self.net = cv2.dnn.readNetFromDarknet(
            config.YOLO_CONFIG, config.YOLO_WEIGHTS
        )

        # Select backend & target
        if config.USE_GPU:
            print("[INFO] Using CUDA GPU backend.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            print("[INFO] Using CPU backend.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [
            layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()
        ]
        print("[INFO] YOLO network loaded successfully.")

    def _generate_colors(self):
        """Generate unique colors for each class."""
        np.random.seed(42)
        self.colors = np.random.randint(
            0, 255, size=(len(self.class_names), 3), dtype="uint8"
        )

    # ── Inference ─────────────────────────────────
    def detect(self, frame):
        """
        Run object detection on a single frame.

        Args:
            frame: BGR image (numpy array).

        Returns:
            list of dicts with keys:
                'class_id', 'label', 'confidence', 'box' (x, y, w, h),
                'color' (B, G, R tuple)
        """
        h, w = frame.shape[:2]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=config.INPUT_SIZE,
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)

        # Forward pass
        layer_outputs = self.net.forward(self.output_layers)

        # Parse detections
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])

                if confidence > config.CONFIDENCE_THRESHOLD:
                    # YOLO returns center (x, y), width, height — relative
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    det_w = int(detection[2] * w)
                    det_h = int(detection[3] * h)

                    # Top-left corner
                    x = int(center_x - det_w / 2)
                    y = int(center_y - det_h / 2)

                    boxes.append([x, y, det_w, det_h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        # Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences,
            config.CONFIDENCE_THRESHOLD,
            config.NMS_THRESHOLD,
        )

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                color = tuple(int(c) for c in self.colors[class_ids[i]])
                results.append({
                    "class_id": class_ids[i],
                    "label": self.class_names[class_ids[i]],
                    "confidence": confidences[i],
                    "box": boxes[i],
                    "color": color,
                })

        return results
