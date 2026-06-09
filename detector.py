import cv2
import numpy as np
import config

class ObjectDetector:
    def __init__(self):
        self._load_class_names()
        self._load_network()
        self._generate_colors()

    def _load_class_names(self):
        with open(config.COCO_NAMES, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"[INFO] Loaded {len(self.class_names)} class labels.")

    def _load_network(self):
        print("[INFO] Loading YOLO network...")
        self.net = cv2.dnn.readNetFromDarknet(config.YOLO_CONFIG, config.YOLO_WEIGHTS)
        if config.USE_GPU:
            print("[INFO] Using CUDA GPU backend.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            print("[INFO] Using CPU backend.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        print("[INFO] YOLO network loaded successfully.")

    def _generate_colors(self):
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype="uint8")

    def _is_in_center(self, box, frame_width, frame_height):
        if not config.CENTER_ONLY_DETECTION:
            return True
        box_center_x = box[0] + box[2] / 2
        box_center_y = box[1] + box[3] / 2
        center_left = frame_width * (1 - config.DETECTION_CENTER_FRACTION) / 2
        center_right = frame_width * (1 + config.DETECTION_CENTER_FRACTION) / 2
        center_top = frame_height * (1 - config.DETECTION_CENTER_FRACTION) / 2
        center_bottom = frame_height * (1 + config.DETECTION_CENTER_FRACTION) / 2
        return (center_left <= box_center_x <= center_right and center_top <= box_center_y <= center_bottom)

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255.0, size=config.INPUT_SIZE, swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)
        boxes = []
        confidences = []
        class_ids = []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])
                if confidence > config.CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    det_w = int(detection[2] * w)
                    det_h = int(detection[3] * h)
                    x = int(center_x - det_w / 2)
                    y = int(center_y - det_h / 2)
                    boxes.append([x, y, det_w, det_h])
                    confidences.append(confidence)
                    class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, config.CONFIDENCE_THRESHOLD, config.NMS_THRESHOLD)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                if self._is_in_center(box, w, h):
                    color = tuple(int(c) for c in self.colors[class_ids[i]])
                    results.append({
                        "class_id": class_ids[i],
                        "label": self.class_names[class_ids[i]],
                        "confidence": confidences[i],
                        "box": box,
                        "color": color,
                    })
        return results