# Smart Parking AI

Detect free and occupied parking spots from a single camera image or video, using YOLOv8 and OpenCV. No sensors and no new hardware: just the cameras a lot already has.

![Example output: occupied slots in red, free slots in green, with a live occupancy panel](detection_result.jpg)

**Built as the ISTE-500/501 Senior Development Project at RIT Dubai by Dilshod, Walid, Sayed and Meiir.**

---

## Features

- Vehicle detection (cars, motorcycles, buses, trucks) with YOLOv8
- Four ways to find parking slots, from fully manual to fully automatic
- Occupancy overlay: green = available, red = occupied, with a stats panel and occupancy bar
- Works on images and on frames sampled from video
- Runs on CPU with the small YOLOv8n model

## Quick start

```bash
git clone https://github.com/ahsayed004/Sayed-Smart-Parking.git
cd Sayed-Smart-Parking

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python smart_parking_simple.py parking_lot.jpg
```

The annotated result is saved as `result.jpg` and shown in a window (press any key to close). To choose a different filename, add `--output`, for example: `python smart_parking_simple.py parking_lot.jpg --output demo-result.jpg`. The default remains `result.jpg`. The YOLOv8 nano weights (`yolov8n.pt`) are included in the repo.

Requires Python 3.9 or newer (developed on 3.12).

## Choosing a detection method

| Script | Slot detection | Manual setup | Best for |
|---|---|---|---|
| `smart_parking_simple.py` | Grid built from detected vehicles | None | Demos and presentations |
| `smart_parking_detection.py` | Hand-drawn slot polygons | Click-to-define GUI | Precise slot boundaries |
| `smart_parking_auto.py` | Hybrid (lines + vehicles) | None | Quick automated analysis |
| `smart_parking_lines.py` | White line-marking detection | None (optional `--debug`) | Lots with visible painted lines |
| `smart_parking_lines_video.py` | Same as `lines`, on video frames | None | Trying the line method on video |

## Contributing

Contributions are welcome, from small fixes to new detection ideas. Good places to start:

- Add automated tests with a few sample images and expected slot counts
- Write a short demo notebook or GIF showing each method side by side
- Support more camera angles (side view, tilted CCTV, fisheye)
- Add a lightweight dashboard or API that serves live occupancy
- Improve slot detection in shadows and low light
- Package the detectors as an installable Python module

To contribute: fork the repo, create a branch, make your change and open a pull request. If you plan something big, open an issue first so we can talk it through.

## License

MIT. See [LICENSE](LICENSE).

---

# Detailed documentation

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` (cv2) | Image processing, drawing, morphological operations |
| `ultralytics` (YOLO) | YOLOv8 object detection model |
| `numpy` | Numerical operations, array handling |
| `scikit-learn` | DBSCAN clustering (used by `smart_parking_lines.py`) |

All scripts default to `yolov8n.pt` in the repository root.

---

## Python files

### 1. `smart_parking_simple.py` (recommended for demos)

**Purpose:** The simplest and most reliable version. Optimized for live presentations and demos where consistent, predictable results matter most.

**How it works:**
1. Detects all vehicles in the image using YOLOv8 (confidence threshold: 0.25)
2. Groups detected vehicles into horizontal rows by clustering their Y-coordinates
3. Calculates the median vehicle size and uses it to define a uniform slot grid (1.25x the median vehicle dimensions)
4. Extends the grid 2 slots beyond the outermost vehicles on each side to show potential empty spaces
5. Checks each slot for occupancy using center-point containment and bounding-box overlap (20% threshold)

**Usage:**
```bash
python smart_parking_simple.py <image_path>
python smart_parking_simple.py parking_lot.jpg --output simple-result.jpg
```

**Output:** Saves the annotated result to `result.jpg` by default, or to the path provided with `--output`. Displays it in a window; press any key to close.

**When to use:** Best choice for presentations and demos. Produces clean, uniform grids that are easy to understand visually.

---

### 2. `smart_parking_detection.py`

**Purpose:** A multi-mode script that supports vehicle detection, manual slot definition, and detection with pre-defined slots.

**Three modes:**

**Mode 1: Vehicle detection (default)**
- Runs YOLOv8 on the input image and draws bounding boxes around all detected vehicles (cars, motorcycles, buses, trucks)
- Displays a stats panel with vehicle count and breakdown by type

**Mode 2: Interactive slot definition (`--define`)**
- Opens the image in a GUI window where you click to define parking slot polygons
- Click 4 corners per slot, then press `n` to confirm
- Keyboard controls:
  - `n`: finish current slot, start next
  - `s`: save all slots to a JSON file
  - `u`: undo last point
  - `r`: reset current slot
  - `q`: quit

**Mode 3: Slot-based detection (`--config`)**
- Loads a previously saved slot configuration (JSON file with polygon coordinates)
- Runs YOLO detection and checks each predefined slot for vehicle overlap (30% threshold)
- Colors slots green (available) or red (occupied) with a semi-transparent overlay

**Usage:**
```bash
# Detect vehicles only
python smart_parking_detection.py parking_lot.jpg

# Interactively define parking slots
python smart_parking_detection.py parking_lot.jpg --define

# Detect using saved slot configuration
python smart_parking_detection.py parking_lot.jpg --config slots.json

# Save a detected image with a custom filename (also works with --config)
python smart_parking_detection.py parking_lot.jpg --output detection-custom.jpg
python smart_parking_detection.py parking_lot.jpg --config slots.json --output configured-result.jpg
```

**Output:** Detection modes save the annotated result to `detection_result.jpg` by default, or to the path provided with `--output`. Interactive `--define` mode saves slot definitions separately and does not use `--output`.

**When to use:** When you need manual control over slot placement, or when automatic slot detection doesn't work well on your lot. Create a `slots.json` once with `--define`, then reuse it for repeated analysis.

---

### 3. `smart_parking_auto.py`

**Purpose:** Automatically detects parking slots without any manual input by analyzing vehicle positions and parking lot line markings.

**How it works:**
1. Detects vehicles using YOLOv8 (confidence threshold: 0.25)
2. Estimates slot dimensions from average detected vehicle sizes (1.25x multiplier)
3. Attempts automatic slot detection using one of three methods:
   - **Lines method:** Canny edge detection + Hough Line Transform to find slot boundary lines, then clusters vertical/horizontal lines into slot regions
   - **Vehicles method:** Groups vehicles into rows by Y-coordinate, estimates slot spacing from vehicle center distances, and generates a slot grid
   - **Hybrid method (default):** Combines both approaches
4. Matches vehicles to slots using bounding-box overlap (20% threshold) and center-point containment
5. Falls back to vehicle-only mode if no slots can be detected

**Usage:**
```bash
python smart_parking_auto.py <image_path>
python smart_parking_auto.py parking_lot.jpg --output auto-result.jpg
```

**Output:** Saves the annotated result to `detection_result.jpg` by default, or to the path provided with `--output`, with a stats panel showing total slots, occupied/available count, and an occupancy bar.

**When to use:** Quick automated analysis where you don't want to define slots by hand. Works best with top-down or near-top-down images where vehicles are arranged in clear rows.

---

### 4. `smart_parking_lines.py`

**Purpose:** The most advanced approach. Detects parking slots by finding white painted line markings on the pavement, then checks for vehicle occupancy.

**How it works:**
1. **White line detection** using three combined methods: HSV color filtering, grayscale thresholding (brightness > 200), and adaptive Gaussian thresholding
2. **Line segment detection** with Canny edge detection followed by a Probabilistic Hough Transform
3. **Line classification** into vertical (slot dividers, 60-120 degrees) and horizontal (row boundaries, 0-30 degrees)
4. **Line clustering** with DBSCAN to merge nearby duplicate detections into single dividers
5. **Slot detection** using four methods and picking the best result: connected components on the inverted line mask, contour analysis, vertical line-based grid generation, and horizontal line-based band detection
6. **Vehicle matching**: YOLO detection (confidence threshold 0.15 for better recall), matched to slots by overlap ratio (15% threshold) and center containment

**Usage:**
```bash
python smart_parking_lines.py <image_path>
python smart_parking_lines.py <image_path> --debug
python smart_parking_lines.py parking_lot.jpg --output lines-result.jpg
```

`--debug` saves intermediate images (`debug_dilated.jpg`, `debug_closed.jpg`, `debug_inverted.jpg`, `debug_cleaned.jpg`) for tuning the line detection pipeline.

**Output:** Saves the annotated result to `detection_result.jpg` by default, or to the path provided with `--output`. The `--debug` option also writes separate intermediate images.

**When to use:** Lots with clearly visible white line markings. Requires `scikit-learn`.

---

### 5. `smart_parking_lines_video.py`

**Purpose:** Runs the line-based detector on frames sampled from a video (for example `CarPark.mp4`). It adds no new detection method: it samples every Nth frame, saves it, and passes it to `LineParkingDetector`.

**Usage:**
```bash
python smart_parking_lines_video.py CarPark.mp4
python smart_parking_lines_video.py CarPark.mp4 --stride 60 --max_frames 10
python smart_parking_lines_video.py CarPark.mp4 --display
python smart_parking_lines_video.py CarPark.mp4 --output experiment
python smart_parking_lines_video.py CarPark.mp4 --output results/test
```

| Option | Default | Description |
|---|---|---|
| `--stride` | 60 | Process every Nth frame |
| `--max_frames` | 5 | Maximum number of frames to process |
| `--display` | off | Show a live window with annotated frames |
| `--output PREFIX` | Video filename stem | Prefix for extracted frames and annotated images |

For video, `--output` is a filename **prefix**, not one output image path. For example, `--output experiment` creates files such as `experiment_frame_00000.jpg` and `experiment_detection_00000.jpg`. With no option, `CarPark.mp4` still produces `CarPark_frame_00000.jpg` and `CarPark_detection_00000.jpg`. You can also use a path prefix such as `--output results/test`; the parent directory is created automatically.

---

## Other files

| File | Description |
|---|---|
| `slots.json` | Example slot configuration with polygon-defined parking spaces (created with `--define` in `smart_parking_detection.py`) |
| `yolov8n.pt` | Pre-trained YOLOv8 nano model weights (COCO dataset) |
| `CarPark.mp4`, `CarPark_frame_00450.jpg` | Sample video and a frame from it |
| `parking_lot.jpg`, `parking_lot_2.jpg`, `sample_parking.jpg`, `sample_parkingv2.jpg` | Sample parking lot images for testing |
| `result.jpg`, `detection_result.jpg` | Example output images |
| `DETECTION_METHODS_DOCUMENTATION.md` | Longer write-up of the detection methods |

## Typical workflow

```bash
# 1. Quick demo
python smart_parking_simple.py parking_lot.jpg

# 2. Precise results: define slots once, reuse them
python smart_parking_detection.py parking_lot.jpg --define
# (click corners, press 'n' after each slot, press 's' to save)
python smart_parking_detection.py parking_lot.jpg --config slots.json

# 3. Fully automatic
python smart_parking_auto.py parking_lot.jpg

# 4. Line-based, for lots with clear white markings
python smart_parking_lines.py parking_lot.jpg --debug
```
