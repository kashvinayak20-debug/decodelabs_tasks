# Project 4: Image & Text Recognition

Two independent computer vision pipelines, both built on **pretrained models** (transfer learning) rather than training a network from scratch: one reads text out of an image (OCR), the other detects and labels objects in an image. A third script is included as a fully self-contained, no-download verification of the object-detection concept.

## Contents

| File | What it does | External download required? |
|---|---|---|
| `project4_ocr.py` | Extracts text from an image using OpenCV preprocessing + Tesseract OCR | Tesseract binary (system-level) |
| `project4_object_detection.py` | Detects & labels 20 object classes using MobileNet-SSD | Yes — 2 model files (~23MB) |
| `project4_face_detection_demo.py` | Verified, no-download face-detection demo (Haar Cascade) | No |

## Setup

1. (Optional but recommended) create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
2. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the Tesseract OCR binary — required by `project4_ocr.py`. `pytesseract` is only a Python wrapper; the actual engine is a separate system install:
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`
   - Windows: [UB-Mannheim Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)
4. **Only if you plan to run `project4_object_detection.py`** — download the pretrained MobileNet-SSD files and place them in this same folder:
   - `MobileNetSSD_deploy.prototxt` (network architecture)
   - `MobileNetSSD_deploy.caffemodel` (trained weights, ~23MB)

   Search "MobileNetSSD_deploy.caffemodel download" — commonly mirrored from the `chuanqi305/MobileNet-SSD` GitHub repo.

`project4_face_detection_demo.py` needs neither external step — it runs immediately after step 2.

## Usage

**OCR** (expects `invoice_sample.jpg` in the same folder, or edit `IMAGE_PATH` to point at your own image):
```bash
python3 project4_ocr.py
```
Prints the extracted text and a confidence-filtered word list to the console, and saves `processed_output.png` — the preprocessed (thresholded) image Tesseract actually read, useful for debugging misreads.

**Object detection** (needs the 2 model files above, plus a test image — default `street_scene.jpg`):
```bash
python3 project4_object_detection.py
```
Saves `detection_output.jpg` with a labeled bounding box drawn for every detection at or above 80% confidence.

**Face detection demo** (no external setup beyond step 2 above):
```bash
python3 project4_face_detection_demo.py
```
Runs against a bundled sample photo (scikit-image's "astronaut" image) and saves `face_detection_output.jpg`.

## Design notes

- All three scripts follow the same shape: **Input** (load + preprocess) → **Process** (run a pretrained model) → **Output** (confidence gate, then report or draw). This is the IPO model used throughout the DecodeLabs curriculum, applied to raw images instead of tabular data.
- The **80% confidence threshold** used across all three scripts is a deliberate floor, not a formality. It's applied identically to OCR word confidences and detection box confidences.
- A confidence gate filters *low-confidence noise* — it does not guarantee correctness. A model can be confidently wrong (see the note in `project4_ocr.py`, where a misread word still clears the threshold). Production systems typically add a second check on top (a dictionary/spell check for OCR, a human review step for anything high-stakes) rather than trusting the raw score alone.
- `project4_object_detection.py` is the script that matches the official spec (real transfer learning via a pretrained CNN). `project4_face_detection_demo.py` exists only as a backup so the same conceptual pipeline can be verified end-to-end without needing the external model download.