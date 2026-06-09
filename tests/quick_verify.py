detector = ObjectDetector()
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from detector import ObjectDetector

detector = ObjectDetector()
print(f"Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
print(f"NMS threshold: {config.NMS_THRESHOLD}")

images = [
    "tests/test_images/dog.jpg",
    "tests/test_images/horses.jpg",
    "tests/test_images/person.jpg",
    "tests/test_images/kite.jpg",
]

for img_path in images:
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"\nCannot read {img_path}")
        continue
    dets = detector.detect(frame)
    labels = [f"{d['label']}({d['confidence']:.2f})" for d in dets]
    unique = set(d["label"] for d in dets)
    print(f"\n{os.path.basename(img_path):15s} -> {len(dets)} detections")
    print(f"  Classes found: {unique}")
    print(f"  Details: {labels}")
