import argparse
import os
import sys
import time

import cv2

import config
from detector import ObjectDetector
from utils import FPSTracker, draw_detections, draw_hud, draw_info_panel
from voice import VoiceAnnouncer


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Object Detection with YOLOv4-tiny & OpenCV")
    parser.add_argument("--source", type=str, default=None,
                        help="Video source: camera index (int) or path to video file. Defaults to config.CAMERA_INDEX.")
    parser.add_argument("--gpu", action="store_true", help="Enable CUDA GPU acceleration (requires OpenCV CUDA build).")
    parser.add_argument("--record", type=str, default=None, help="Path to save output video (e.g., output.avi).")
    return parser.parse_args()


def resolve_source(source_arg):
    if source_arg is None:
        return config.CAMERA_INDEX
    try:
        return int(source_arg)
    except ValueError:
        if os.path.isfile(source_arg):
            return source_arg
        print(f"[ERROR] Video file not found: {source_arg}")
        sys.exit(1)


def main():
    args = parse_args()

    if args.gpu:
        config.USE_GPU = True

    for path, name in [
        (config.YOLO_WEIGHTS, "YOLO weights"),
        (config.YOLO_CONFIG, "YOLO config"),
        (config.COCO_NAMES, "COCO names"),
    ]:
        if not os.path.isfile(path):
            print(f"[ERROR] Missing {name}: {path}")
            print("[INFO]  Run `python download_models.py` to fetch model files.")
            sys.exit(1)

    detector = ObjectDetector()
    announcer = VoiceAnnouncer()

    source = resolve_source(args.source)
    print(f"[INFO] Opening video source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("[ERROR] Cannot open video source.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Capture resolution: {actual_w}x{actual_h}")

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.record, fourcc, 25.0, (actual_w, actual_h))
        print(f"[INFO] Recording to: {args.record}")

    fps_tracker = FPSTracker()
    paused = False
    screenshot_dir = os.path.join(config.BASE_DIR, "screenshots")

    print("[INFO] Detection started. Press 'q' to quit.\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video stream.")
                break

            detections = detector.detect(frame)
            announcer.update(detections, actual_w, actual_h)

            frame = draw_detections(frame, detections)
            fps = fps_tracker.update()
            frame = draw_hud(frame, fps, len(detections))
            frame = draw_info_panel(frame, detections)

            if writer is not None:
                writer.write(frame)

        cv2.imshow(config.WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quit requested.")
            break
        elif key == ord("s"):
            os.makedirs(screenshot_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(screenshot_dir, f"screenshot_{timestamp}.jpg")
            cv2.imwrite(path, frame)
            print(f"[INFO] Screenshot saved: {path}")
        elif key == ord("p"):
            paused = not paused
            state = "PAUSED" if paused else "RUNNING"
            print(f"[INFO] {state}")
        elif key == ord("+") or key == ord("="):
            config.CONFIDENCE_THRESHOLD = min(config.CONFIDENCE_THRESHOLD + 0.05, 0.95)
            print(f"[INFO] Confidence threshold: {config.CONFIDENCE_THRESHOLD:.2f}")
        elif key == ord("-"):
            config.CONFIDENCE_THRESHOLD = max(config.CONFIDENCE_THRESHOLD - 0.05, 0.10)
            print(f"[INFO] Confidence threshold: {config.CONFIDENCE_THRESHOLD:.2f}")
        elif key == ord("v"):
            config.VOICE_ENABLED = not config.VOICE_ENABLED
            state = "ON" if config.VOICE_ENABLED else "OFF"
            print(f"[INFO] Voice announcements: {state}")

    announcer.shutdown()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released. Goodbye!")


if __name__ == "__main__":
    main()
