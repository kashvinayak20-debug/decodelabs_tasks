"""
================================================================
 Project 4 -- Path 2 BONUS: a fully self-contained detection demo


 Why this file exists: MobileNetSSD_deploy.caffemodel (~23MB) has to
 be downloaded manually (see project4_object_detection.py's header).
 This script proves the *same conceptual pipeline* -- pretrained
 model -> preprocess -> detect -> confidence gate -> draw boxes --
 end-to-end, using a Haar Cascade classifier that ships FREE inside
 opencv-python (no download needed), tested against a real photo.

 Haar Cascades are an older, classical (non-deep-learning) detection
 technique -- included here only so you have one fully verified,
 runnable result. The deep-learning version is the one that matches
 the deck's spec; treat this as a working stand-in, not a replacement.
================================================================
"""

import cv2
from skimage import data
from skimage import io as skio

OUTPUT_PATH = "face_detection_output.jpg"
SCALE_FACTOR = 1.1     # how much the image shrinks at each scale check
MIN_NEIGHBORS = 5      # higher = fewer false positives, may miss faint faces


def load_test_image():
    """Uses scikit-image's bundled sample photo (a real photograph, not synthetic shapes)."""
    rgb_image = data.astronaut()             # real photo, ships with scikit-image
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image


def load_model():
    """PHASE 1: load a pretrained classifier -- bundled with opencv-python, no download needed."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def detect_faces(model, image):
    """PHASE 2 (PROCESS): classical sliding-window detection over grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(
        gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=(30, 30)
    )
    return faces


def draw_boxes(image, faces):
    """PHASE 3 (OUTPUT): annotate every accepted detection."""
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "face", (x, max(y - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image


def main():
    print("PROJECT 4 -- BONUS: VERIFIED DETECTION DEMO (Haar Cascade)")

    image = load_test_image()
    model = load_model()
    faces = detect_faces(model, image)

    print(f"Faces detected: {len(faces)}")
    for i, (x, y, w, h) in enumerate(faces):
        print(f"  face {i+1}: box=({x},{y},{w},{h})")

    annotated = draw_boxes(image.copy(), faces)
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"Annotated image saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
