# =============================================================================
# test.py — Testing / Recognition Pipeline for PCA + ANN Face Recognition
# =============================================================================
#
# Run this file after training:
#       python test.py
#
# Modes:
#   1. BATCH MODE   — test on a folder of images at once
#   2. SINGLE MODE  — test one image file (passed as command-line argument)
#
# Usage:
#   python test.py                         → batch test on dataset/test_images/
#   python test.py path/to/face.jpg        → single image test
#   python test.py path/to/imposter.jpg    → imposter detection
# =============================================================================

import os
import sys
import numpy as np
import cv2

from utils import (load_model, preprocess_single_image,
                   display_result, IMAGE_SIZE, DATASET_PATH)


# =============================================================================
# 1. RECOGNISE A SINGLE IMAGE
# =============================================================================
def recognize_face(image_path, model_data, show_window=True):
    """
    Recognises the person in a single image file.

    Pipeline:
        Raw Image → Grayscale → Resize → Flatten
        → Mean Subtraction → PCA Projection → ANN Prediction

    Args:
        image_path : path to the face image file
        model_data : dictionary loaded from saved_model.pkl
        show_window: if True, opens an OpenCV display window

    Returns:
        (predicted_name, confidence)
    """
    pca   = model_data['pca']
    ann   = model_data['ann']
    names = model_data['names']

    # ── Step 1: Preprocess (grayscale, resize, flatten) ───────────────
    print(f"\n[TEST] Processing: {image_path}")
    face_vector = preprocess_single_image(image_path, IMAGE_SIZE)
    # face_vector shape: (mn,)   e.g. (10000,) for 100×100 images

    # ── Step 2: Project onto eigenface space ──────────────────────────
    # Mean is subtracted inside transform_single
    signature = pca.transform_single(face_vector)
    # signature shape: (K,)

    # ── Step 3: ANN Prediction with imposter detection ─────────────────
    name, confidence = ann.predict_single(signature)

    # ── Step 4: Display result ─────────────────────────────────────────
    status = "RECOGNISED" if name != "IMPOSTER" else "⚠  IMPOSTER DETECTED"
    print(f"[TEST] Result     : {name}")
    print(f"[TEST] Confidence : {confidence * 100:.1f}%")
    print(f"[TEST] Status     : {status}")

    if show_window:
        display_result(image_path, name, confidence * 100)

    return name, confidence


# =============================================================================
# 2. BATCH TESTING ON A FOLDER
# =============================================================================
def batch_test(test_folder, model_data):
    """
    Tests all images inside a folder and prints a summary table.

    The test folder should contain sub-folders named by person:
        test_images/
            person1/  ← known (enrolled) images
            person2/  ← known images
            imposters/← unknown faces to test imposter detection

    Args:
        test_folder : path to folder containing test images
        model_data  : loaded model dictionary
    """
    if not os.path.exists(test_folder):
        print(f"[WARN] Test folder '{test_folder}' not found.")
        print("       Creating a demo test using the training dataset instead.")
        _demo_test(model_data)
        return

    print("\n" + "=" * 60)
    print("   BATCH TESTING")
    print("=" * 60)

    results     = []
    correct     = 0
    total       = 0
    imposter_ok = 0
    imposter_total = 0

    for label_name in sorted(os.listdir(test_folder)):
        folder = os.path.join(test_folder, label_name)
        if not os.path.isdir(folder):
            continue

        is_imposter_folder = label_name.lower() in ('imposters', 'imposter', 'unknown')

        for img_file in sorted(os.listdir(folder)):
            img_path = os.path.join(folder, img_file)
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue

            try:
                name, conf = recognize_face(img_path, model_data, show_window=False)
            except Exception as e:
                print(f"[ERROR] {img_path}: {e}")
                continue

            if is_imposter_folder:
                imposter_total += 1
                if name == "IMPOSTER":
                    imposter_ok += 1
                    status = "✓ CORRECTLY REJECTED"
                else:
                    status = f"✗ WRONGLY IDENTIFIED AS {name}"
            else:
                total += 1
                if name == label_name:
                    correct += 1
                    status = "✓ CORRECT"
                else:
                    status = f"✗ WRONG (predicted: {name})"

            results.append((label_name, img_file, name, f"{conf*100:.1f}%", status))

    # ── Print Results Table ────────────────────────────────────────────
    print(f"\n{'True Label':<20} {'Image':<25} {'Predicted':<20} {'Conf':>6}  Status")
    print("-" * 90)
    for row in results:
        print(f"{row[0]:<20} {row[1]:<25} {row[2]:<20} {row[3]:>6}  {row[4]}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Known faces accuracy     : {correct}/{total} = "
          f"{correct/total*100:.1f}%" if total > 0 else "  No known faces tested.")
    if imposter_total > 0:
        print(f"  Imposter rejection rate  : {imposter_ok}/{imposter_total} = "
              f"{imposter_ok/imposter_total*100:.1f}%")
    print("=" * 60)


# =============================================================================
# 3. DEMO TEST — uses training images when no separate test folder exists
# =============================================================================
def _demo_test(model_data):
    """
    Fallback demo: picks a few images from the dataset and tests them.
    Useful when no separate test_images/ folder is available.
    """
    from utils import load_dataset
    print("\n[DEMO] Running demo test on dataset images ...")

    images, labels, names = load_dataset()
    pca = model_data['pca']
    ann = model_data['ann']

    # Project all images
    sigs = pca.transform(images)

    correct = 0
    print(f"\n{'True':<20} {'Predicted':<20} {'Conf':>8}")
    print("-" * 52)
    for i in range(min(20, len(images))):          # show first 20
        name, conf = ann.predict_single(sigs[i])
        true_name  = names[labels[i]]
        mark = "✓" if name == true_name else "✗"
        print(f"{true_name:<20} {name:<20} {conf*100:>7.1f}%  {mark}")
        if name == true_name:
            correct += 1

    n = min(20, len(images))
    print(f"\nDemo accuracy: {correct}/{n} = {correct/n*100:.1f}%")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Load the saved model
    model_data = load_model()

    if len(sys.argv) > 1:
        # ── Single image mode ─────────────────────────────────────────
        image_path = sys.argv[1]
        recognize_face(image_path, model_data, show_window=True)

    else:
        # ── Batch mode ────────────────────────────────────────────────
        # Default test folder: dataset/test_images/
        test_folder = os.path.join(DATASET_PATH, "test_images")
        batch_test(test_folder, model_data)
