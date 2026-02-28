# Real-Time Object Detection with YOLOv4-tiny + OpenCV

This project uses your **webcam** (or any video file) to detect objects in real time. It can recognise **80 different types of objects** — people, cars, dogs, cats, chairs, phones, bottles, and many more.

It uses a small and fast AI model called **YOLOv4-tiny** which runs through **OpenCV** (a popular computer vision library). Everything runs **locally on your computer** — no internet needed after setup.

---

## What Does This Project Do?

In simple words:

1. **Opens your webcam** and starts reading video frames (pictures) one by one.
2. **Sends each frame to the YOLO AI model**, which looks at the picture and finds objects.
3. **Draws boxes around each object** it finds, along with the object name and how confident it is.
4. **Speaks the name of new objects** that appear in the centre of the frame (confidence > 60%).
5. **Shows you the result** in a live video window on your screen.
6. You can **pause, take screenshots, adjust sensitivity, toggle voice**, and **quit** using keyboard keys.

---

## Flow Diagram

Below is a visual flow of how the program works from start to finish:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PROGRAM START                                │
│                     (python main.py)                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  Check if model files │
               │  exist in /models     │
               └───────────┬───────────┘
                           │
                    ┌──────┴──────┐
                    │  Missing?   │
                    └──────┬──────┘
                     YES   │   NO
                ┌──────────┤──────────┐
                ▼                     ▼
        ┌──────────────┐   ┌──────────────────┐
        │ Show error:  │   │ Load YOLO model  │
        │ "Run         │   │ (weights + config│
        │ download_    │   │  + class names)  │
        │ models.py"   │   └────────┬─────────┘
        │ then EXIT    │            │
        └──────────────┘            ▼
                           ┌──────────────────┐
                           │ Open webcam or   │
                           │ video file       │
                           └────────┬─────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │      MAIN LOOP START      │◄──────────────┐
                    └─────────────┬─────────────┘               │
                                  │                             │
                                  ▼                             │
                    ┌───────────────────────────┐               │
                    │  Read one frame from      │               │
                    │  webcam / video           │               │
                    └─────────────┬─────────────┘               │
                                  │                             │
                                  ▼                             │
                    ┌───────────────────────────┐               │
                    │  Convert frame to "blob"  │               │
                    │  (resize to 416×416,      │               │
                    │   normalise colours)      │               │
                    └─────────────┬─────────────┘               │
                                  │                             │
                                  ▼                             │
                    ┌───────────────────────────┐               │
                    │  Feed blob into YOLO      │               │
                    │  neural network           │               │
                    │  (forward pass)           │               │
                    └─────────────┬─────────────┘               │
                                  │                             │
                                  ▼                             │
                    ┌───────────────────────────┐               │
                    │  Filter detections:       │               │
                    │  • Keep only confidence   │               │
                    │    above threshold (25%)  │               │
                    │  • Remove duplicate boxes │               │
                    │    (NMS)                  │               │
                    └─────────────┬─────────────┘               │
                                  │                             │
                                  ▼                             │
                    ┌───────────────────────────┐               │
                    │  VOICE CHECK              │               │
                    │  For each detection:      │               │
                    │  • Is label NEW?          │               │
                    │  • Confidence > 60%?      │               │
                    │  • In centre of frame?    │               │
                    │  • Cooldown expired? (3s) │               │
                    │  If all YES → speak name  │               │
                    │  (background thread)      │               │
                    └─────────────┬─────────────┘               │
                                  │                             │
                                  ▼                             │
                    ┌───────────────────────────┐               │
                    │  Draw on the frame:       │               │
                    │  • Bounding boxes         │               │
                    │  • Labels + confidence %  │               │
                    │  • FPS counter            │               │
                    │  • Object count panel     │               │
                    └─────────────┬─────────────┘               │
                                  │                             │
                                  ▼                             │
                    ┌───────────────────────────┐               │
                    │  Show frame in window     │               │
                    └─────────────┬─────────────┘               │
                                  │                             │
                                  ▼                             │
                    ┌───────────────────────────┐               │
                    │  Check keyboard input     │               │
                    │  q = quit                 │               │
                    │  s = screenshot           │               │
                    │  p = pause/resume         │               │
                    │  v = toggle voice on/off  │               │
                    │  +/- = adjust threshold   │               │
                    └─────────────┬─────────────┘               │
                                  │                             │
                           ┌──────┴──────┐                      │
                           │ q pressed?  │                      │
                           └──────┬──────┘                      │
                            YES   │   NO                        │
                   ┌──────────────┤                             │
                   ▼              └─────────────────────────────┘
          ┌──────────────────┐
          │  Release webcam  │
          │  Close window    │
          │  EXIT            │
          └──────────────────┘
```

---

## Project Structure

```
object detection/
│
├── main.py               ← Entry point. Runs the detection loop.
├── detector.py           ← Loads the YOLO model and runs detection.
├── utils.py              ← Draws boxes, FPS counter, info panel.
├── voice.py              ← Voice announcer (pyttsx3, background thread).
├── config.py             ← All settings (thresholds, camera, voice, paths).
├── download_models.py    ← Downloads YOLO model files (run once).
├── requirements.txt      ← Python packages needed.
├── README.md             ← This file.
├── .gitignore            ← Files to ignore in git.
│
├── venv/                 ← Virtual environment (created during setup).
│
├── models/               ← Model files (created by download_models.py).
│   ├── yolov4-tiny.weights   (23 MB — the trained AI brain)
│   ├── yolov4-tiny.cfg       (network architecture)
│   └── coco.names            (list of 80 object names)
│
└── screenshots/          ← Saved screenshots (created when you press 's').
```

### What Each File Does

| File | What It Does |
|---|---|
| **main.py** | The main program. Opens the camera, runs detection in a loop, shows results, handles keyboard input. |
| **detector.py** | Contains the `ObjectDetector` class. Loads the YOLO model, converts frames to the right format, runs the AI, and filters results. |
| **utils.py** | Helper functions to draw bounding boxes, the FPS counter, and the info panel on the video frame. Also has the `FPSTracker` class. |
| **voice.py** | Voice announcer using pyttsx3. Speaks new object names in a background thread. Has cooldown, centre-frame check, and confidence filter. |
| **config.py** | All settings in one place — camera index, confidence threshold, input size, voice settings, window name, etc. Change settings here. |
| **download_models.py** | A script that downloads the 3 required model files from the internet. You only need to run this once. |

---

## Setup Guide (Step by Step)

### Prerequisites

- **Python 3.8 or newer** installed ([download here](https://www.python.org/downloads/))
- A **webcam** (built-in or USB) — or a video file

### Step 1: Open a terminal in the project folder

```bash
cd "C:\Users\HP\Desktop\object detection"
```

### Step 2: Create a virtual environment

A virtual environment keeps this project's packages separate from your system Python.

```bash
python -m venv venv
```

### Step 3: Activate the virtual environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

You should see `(venv)` at the start of your terminal prompt.

### Step 4: Install the required packages

```bash
pip install -r requirements.txt
```

This installs:
- **opencv-contrib-python** — computer vision library with GUI support
- **numpy** — fast number crunching for arrays
- **pyttsx3** — offline text-to-speech engine for voice announcements

### Step 5: Download the AI model files

```bash
python download_models.py
```

This downloads 3 files (~23 MB total) into the `models/` folder:
- `yolov4-tiny.weights` — the trained neural network weights
- `yolov4-tiny.cfg` — the network architecture
- `coco.names` — the list of 80 object class names

### Step 6: Run the detector!

```bash
python main.py
```

A window will open showing your webcam feed with detected objects highlighted.

---

## Keyboard Controls

| Key | What It Does |
|-----|---|
| **q** | Quit the program |
| **s** | Save a screenshot to the `screenshots/` folder |
| **p** | Pause or resume the video |
| **v** | Toggle voice announcements ON / OFF |
| **+** (or **=**) | Make detection stricter (increase confidence threshold by 5%) |
| **-** | Make detection more sensitive (decrease confidence threshold by 5%) |

---

## Configuration

All settings are in **config.py**. Here are the important ones:

| Setting | Default | What It Means |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.25` | How sure the AI must be before showing a detection. Higher = fewer but more accurate boxes. |
| `NMS_THRESHOLD` | `0.45` | Controls how much overlap is allowed between boxes. Lower = fewer duplicate boxes. |
| `INPUT_SIZE` | `(416, 416)` | The size the image is resized to before feeding to the AI. Smaller = faster but less accurate. |
| `CAMERA_INDEX` | `0` | Which camera to use. `0` = default webcam, `1` = second camera, etc. |
| `VIDEO_WIDTH` | `1280` | Requested webcam resolution width. |
| `VIDEO_HEIGHT` | `720` | Requested webcam resolution height. |
| `USE_GPU` | `False` | Set to `True` if your OpenCV is built with CUDA support (for NVIDIA GPUs). |
| `VOICE_ENABLED` | `True` | Enable or disable voice announcements. |
| `VOICE_CONFIDENCE_THRESHOLD` | `0.60` | Minimum confidence for voice (higher than visual threshold). |
| `VOICE_COOLDOWN` | `3.0` | Seconds before the same object name can be spoken again. |
| `VOICE_CENTER_FRACTION` | `0.5` | Size of the centre zone (0.5 = middle 50% of the frame). |
| `VOICE_RATE` | `175` | Speech speed in words per minute. |
| `VOICE_VOLUME` | `1.0` | Speaker volume (0.0 to 1.0). |

### Use a video file instead of webcam

In `main.py`, run:
```bash
python main.py --source "path/to/your/video.mp4"
```

### Record the output

```bash
python main.py --record output.avi
```

---

## How YOLOv4-tiny Works (Simple Explanation)

**YOLO** stands for **"You Only Look Once"**. Unlike older methods that scan an image multiple times, YOLO looks at the entire image **in one pass** and predicts all objects at once. That's why it's so fast.

**YOLOv4-tiny** is a smaller version of YOLOv4 — it uses fewer layers in the neural network, which makes it:
- **Faster** (30+ FPS on a regular CPU)
- **Slightly less accurate** than the full model (but still very good)

### The Detection Pipeline:

1. **Input**: Your webcam frame (e.g. 1280×720 pixels)
2. **Resize**: Shrink to 416×416 pixels and normalise pixel values from 0–255 to 0.0–1.0
3. **Neural Network**: The image passes through ~30 convolutional layers that extract features (edges, shapes, textures)
4. **Output**: The network outputs a grid of predictions. Each grid cell predicts bounding boxes, confidence scores, and class probabilities.
5. **Filtering**: Remove low-confidence detections (below threshold)
6. **NMS (Non-Maximum Suppression)**: If multiple boxes overlap on the same object, keep only the best one
7. **Result**: Clean list of objects with their positions, names, and confidence percentages

---

## What Objects Can It Detect?

The model is trained on the **COCO dataset** and can detect these **80 classes**:

> person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, sofa, potted plant, bed, dining table, toilet, TV monitor, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

---

## Performance Tips

| Goal | What to Change |
|---|---|
| **Faster detection** | Set `INPUT_SIZE = (320, 320)` in config.py |
| **More accurate detection** | Set `INPUT_SIZE = (608, 608)` in config.py |
| **Even more accuracy** | Use the full YOLOv4 model (uncomment in download_models.py, ~245 MB) |
| **GPU acceleration** | Build OpenCV with CUDA and set `USE_GPU = True` |

---

## Voice Announcements (How It Works)

When the program detects an object, it checks 4 conditions before speaking:

1. **New object** — The label must not have been present in the previous frame. This prevents it from repeating "person person person" every frame.
2. **High confidence** — The detection confidence must be above 60% (configurable via `VOICE_CONFIDENCE_THRESHOLD`). This avoids speaking false detections.
3. **Centre of frame** — The object's centre must be within the middle 50% of the screen (configurable via `VOICE_CENTER_FRACTION`). This means only objects you're pointing the camera at get announced, not things at the edges.
4. **Cooldown** — Each label has a 3-second cooldown. Even if an object disappears and reappears, it won't be spoken again until 3 seconds have passed.

Speech runs in a **background thread** so it never slows down the video. If the previous announcement is still playing, new ones are skipped to avoid queue build-up.

```
  Detection comes in → Is label NEW? ──NO──→ Skip
                           │
                          YES
                           │
                           ▼
                   Confidence > 60%? ──NO──→ Skip
                           │
                          YES
                           │
                           ▼
                   In centre frame? ──NO──→ Skip
                           │
                          YES
                           │
                           ▼
                   Cooldown expired? ──NO──→ Skip
                           │
                          YES
                           │
                           ▼
                    SPEAK object name
                   (background thread)
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Make sure the venv is activated and run `pip install -r requirements.txt` |
| `cv2.error: The function is not implemented` | Use `opencv-contrib-python` (not `opencv-python`). Already set in requirements.txt. |
| `Cannot open video source` | Check your webcam is connected. Try changing `CAMERA_INDEX` in config.py. |
| Model files missing | Run `python download_models.py` |
| Low FPS | Reduce `INPUT_SIZE` to `(320, 320)` in config.py |
| Too many false detections | Increase `CONFIDENCE_THRESHOLD` in config.py (e.g. to 0.6) |
| Missing real objects | Decrease `CONFIDENCE_THRESHOLD` (e.g. to 0.3) |
| Voice not working | Check speakers are on. Run `python -c "import pyttsx3; e=pyttsx3.init(); e.say('test'); e.runAndWait()"` |
| Voice too fast/slow | Change `VOICE_RATE` in config.py (default 175) |
| Voice too chatty | Increase `VOICE_COOLDOWN` or `VOICE_CONFIDENCE_THRESHOLD` |
| Want to disable voice | Press `v` at runtime, or set `VOICE_ENABLED = False` in config.py |

---

## License

This project uses the YOLOv4 model by Alexey Bochkovskiy et al. The model weights are released under the [MIT License](https://github.com/AlexeyAB/darknet/blob/master/LICENSE). This project code is provided for educational purposes.
