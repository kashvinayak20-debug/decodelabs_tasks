"""
================================================================
 Project 4 -- Path 1: Optical Character Recognition (OCR)


 Pipeline : IPO Model -> Input (raw image) | Process (preprocess +
            pretrained OCR engine) | Output (validated text)
 Engine   : Tesseract OCR (via pytesseract), preceded by an OpenCV
            preprocessing stage.
================================================================
"""

import cv2
import pytesseract
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

IMAGE_PATH = "invoice_sample.jpg"
CONFIDENCE_THRESHOLD = 80          # % -- discard low-confidence guesses
PSM_MODE = 6                       # assume a single uniform block of text


def preprocess(image_path):
    """
    PHASE 1 (INPUT): clean the raw image before the OCR engine ever
    sees it. Every step below removes one specific kind of ambiguity.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Grayscale: Tesseract reads shapes, not colors -- color is
    # irrelevant (and sometimes actively confusing) information here.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur: smooths sensor noise / JPEG artifacts *before*
    # thresholding, so noise doesn't get amplified into false edges.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's thresholding: forces every pixel to pure black or white.
    # Otsu is *adaptive* -- it finds the best cutoff automatically
    # instead of using one fixed brightness value for every image.
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return image, binary


def run_ocr(binary_image):
    """
    PHASE 2 (PROCESS): run the pretrained Tesseract engine.
    psm 6 = "assume a single uniform block of text" -- the right mode
    for something like an invoice body. (psm 7 = one line, e.g. a
    license plate; psm 11 = sparse scattered text.)
    """
    config = f"--psm {PSM_MODE}"
    text = pytesseract.image_to_string(binary_image, config=config)
    data = pytesseract.image_to_data(binary_image, config=config, output_type=Output.DICT)
    return text, data


def apply_confidence_gate(data, threshold=CONFIDENCE_THRESHOLD):
    """
    PHASE 3 (OUTPUT): keep only words Tesseract itself is confident
    about. This does NOT guarantee correctness -- it filters out
    low-confidence noise, but a high-confidence wrong answer still
    passes through. See the README note below.
    """
    kept = []
    for i, word in enumerate(data["text"]):
        conf = int(data["conf"][i]) if data["conf"][i] not in ("", "-1") else -1
        if word.strip() and conf >= threshold:
            kept.append((word, conf))
    return kept


def main():
    print("PROJECT 4 -- PATH 1: OCR PIPELINE")


    image, processed = preprocess(IMAGE_PATH)
    print(f"Image shape: {image.shape}")

    text, data = run_ocr(processed)
    print("\n=== EXTRACTED TEXT ===")
    print(text)

    kept_words = apply_confidence_gate(data)
    print(f"=== WORDS >= {CONFIDENCE_THRESHOLD}% CONFIDENCE ===")
    for word, conf in kept_words:
        print(f"  {word:<20} confidence: {conf}%")

    cv2.imwrite("processed_output.png", processed)
    print("\nProcessed (thresholded) image saved -> processed_output.png")

    # NOTE ON THE CONFIDENCE GATE:
    # A high confidence score means Tesseract is confident about its
    # *guess*, not that the guess is correct. Capital D is visually
    # close to O in many fonts -- a misread like this can still score
    # 90%+. Production OCR pipelines usually add a second check on top
    # of the raw score (a dictionary/spell check, or a human review
    # step for anything financial/legal) rather than trusting
    # confidence alone.


if __name__ == "__main__":
    main()
