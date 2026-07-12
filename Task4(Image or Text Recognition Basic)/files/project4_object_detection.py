"""
================================================================
 Project 4 -- Path 2: Object Detection (deep learning, transfer learning)
 

 Pipeline : IPO Model -> Input (raw image) | Process (preprocess +
            pretrained CNN) | Output (validated bounding boxes)
 Model    : MobileNet-SSD (Single Shot Detector), trained on the
            PASCAL VOC dataset (21 classes incl. background).

 SETUP (one-time, do this yourself before running):
 This script uses TRANSFER LEARNING -- it loads a network someone
 else already trained, rather than training one from scratch. That
 means it needs 2 external files placed in this same folder:
   1. MobileNetSSD_deploy.prototxt      (the network architecture)
   2. MobileNetSSD_deploy.caffemodel    (the trained weights, ~23MB)
 Search "MobileNetSSD_deploy.caffemodel download" -- these are widely
 mirrored (e.g. the chuanqi305/MobileNet-SSD GitHub repo). They could
 not be downloaded automatically in this sandboxed environment (the
 usual hosts are outside its network allowlist), so this one step is
 left for you to do locally.
================================================================
"""

import cv2
import numpy as np

PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"
IMAGE_PATH = "test_photo.jpg"
CONFIDENCE_THRESHOLD = 0.8   # 80% -- same "confidence gate" concept as the OCR path

# The 21 classes MobileNet-SSD was trained to recognize (PASCAL VOC)
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
]
np.random.seed(42)
BOX_COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def load_model():
    """PHASE 1: load the pretrained network (transfer learning -- no training happens here)."""
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    return net


def preprocess(image):
    """
    PHASE 2 (INPUT prep): build a 'blob' the network can read.
    - resize to 300x300: the fixed input size this architecture expects
    - scale by 1/127.5 and subtract 127.5: matches the exact pixel
      normalization the network was TRAINED on -- skipping this
      silently produces garbage predictions, since the network never
      learned to interpret raw 0-255 pixel values.
    """
    resized = cv2.resize(image, (300, 300))
    blob = cv2.dnn.blobFromImage(resized, 0.007843, (300, 300), 127.5)
    return blob


def detect(net, blob):
    """PHASE 3 (PROCESS): one forward pass -> every candidate box + class score at once (Single Shot)."""
    net.setInput(blob)
    detections = net.forward()
    return detections


def draw_detections(image, detections, threshold=CONFIDENCE_THRESHOLD):
    """PHASE 4 (OUTPUT): confidence gate + coordinate decoding + annotation."""
    h, w = image.shape[:2]
    kept = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < threshold:
            continue  # confidence gate: discard low-confidence boxes

        class_id = int(detections[0, 0, i, 1])
        # box coords come out as fractions of image size (0-1) -> convert to pixels
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")

        label = f"{CLASSES[class_id]}: {confidence * 100:.1f}%"
        color = BOX_COLORS[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        kept += 1

    print(f"Kept {kept} detections at >= {threshold*100:.0f}% confidence")
    return image


def main():
    print("PROJECT 4 -- PATH 2: OBJECT DETECTION (MobileNet-SSD)")

    net = load_model()
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    blob = preprocess(image)
    detections = detect(net, blob)
    annotated = draw_detections(image, detections)

    cv2.imwrite("detection_output.jpg", annotated)
    print("Annotated image saved -> detection_output.jpg")


if __name__ == "__main__":
    main()
