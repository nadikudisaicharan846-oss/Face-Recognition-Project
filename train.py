# =============================================================================
# train.py — Training Pipeline for PCA + ANN Face Recognition
# =============================================================================
#
# Run this file first:
#       python train.py
#
# What it does:
#   1. Loads all face images from the 'dataset/' folder
#   2. Applies PCA to extract eigenfaces and signatures
#   3. Splits data into 60% training / 40% testing
#   4. Trains an ANN (MLPClassifier) on the PCA signatures
#   5. Evaluates accuracy for multiple K values and plots a graph
#   6. Saves the best model to disk (saved_model.pkl)
#   7. Visualises and saves eigenfaces
# =============================================================================

import numpy as np
from sklearn.model_selection import train_test_split

from utils import (load_dataset, show_eigenfaces,
                   plot_accuracy_vs_k, save_model)
from pca  import SklearnPCA       # Change to ManualPCA for manual implementation
from ann  import ANNClassifier


# =============================================================================
# CONFIGURATION — tweak these values to experiment
# =============================================================================

# Range of K values to test (number of principal components)
K_VALUES = [5, 10, 20, 30, 50, 75, 100]

# Fixed split: 60% train, 40% test (as specified in the project)
TEST_SIZE    = 0.40
RANDOM_SEED  = 42

# Imposter detection threshold (lower = stricter)
IMPOSTER_THRESHOLD = 0.50


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def train():
    print("=" * 65)
    print("   PCA + ANN Face Recognition — TRAINING PHASE")
    print("=" * 65)

    # ── 1. LOAD DATASET ────────────────────────────────────────────────
    print("\n[STEP 1] Loading dataset ...")
    images, labels, names = load_dataset()
    print(f"         Image matrix shape : {images.shape}")
    print(f"         Persons enrolled   : {names}")

    num_classes = len(names)
    if num_classes < 2:
        print("[ERROR] Need at least 2 persons in the dataset.")
        return

    # ── 2. SPLIT DATA: 60% TRAIN / 40% TEST ───────────────────────────
    print("\n[STEP 2] Splitting data (60% train / 40% test) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size    = TEST_SIZE,
        stratify     = labels,       # ensures proportional class split
        random_state = RANDOM_SEED
    )
    print(f"         Train samples : {len(X_train)}")
    print(f"         Test samples  : {len(X_test)}")

    # ── 3. EXPERIMENT OVER DIFFERENT K VALUES ─────────────────────────
    print("\n[STEP 3] Testing different K values ...\n")
    print(f"{'K':>6} | {'Accuracy':>10}")
    print("-" * 20)

    k_values   = []
    accuracies = []
    best_acc   = -1
    best_k     = None
    best_model_data = None

    for k in K_VALUES:
        # Cap K at the number of training samples
        k_actual = min(k, len(X_train) - 1)

        # ── 3a. Apply PCA ─────────────────────────────────────────────
        pca = SklearnPCA(k=k_actual)
        X_train_pca = pca.fit_transform(X_train)   # shape (n_train, k_actual)
        X_test_pca  = pca.transform(X_test)         # shape (n_test,  k_actual)

        # ── 3b. Train ANN ─────────────────────────────────────────────
        ann = ANNClassifier()
        ann.imposter_threshold = IMPOSTER_THRESHOLD
        ann.train(X_train_pca, y_train, names)

        # ── 3c. Evaluate on test set ──────────────────────────────────
        acc = ann.evaluate(X_test_pca, y_test, show_report=False)

        print(f"{k_actual:>6} | {acc:>9.2f}%")

        k_values.append(k_actual)
        accuracies.append(acc)

        # Keep the best model
        if acc > best_acc:
            best_acc = acc
            best_k   = k_actual
            best_model_data = {
                'ann'        : ann,
                'pca'        : pca,
                'mean_face'  : pca.pca.mean_,      # mean face vector
                'eigenfaces' : pca.eigenfaces,      # (K, mn)
                'names'      : names,
                'k'          : k_actual
            }

    # ── 4. REPORT BEST RESULT ──────────────────────────────────────────
    print("\n" + "=" * 40)
    print(f"  BEST K = {best_k}  |  Accuracy = {best_acc:.2f}%")
    print("=" * 40)

    # ── 5. VISUALISE EIGENFACES ───────────────────────────────────────
    print("\n[STEP 4] Visualising eigenfaces ...")
    show_eigenfaces(best_model_data['eigenfaces'], num_faces=10)

    # ── 6. PLOT ACCURACY vs K GRAPH ───────────────────────────────────
    print("\n[STEP 5] Plotting Accuracy vs K graph ...")
    plot_accuracy_vs_k(k_values, accuracies)

    # ── 7. FULL EVALUATION WITH BEST K ────────────────────────────────
    print("\n[STEP 6] Full evaluation with best K ...")
    best_ann = best_model_data['ann']
    best_pca = best_model_data['pca']

    X_test_best = best_pca.transform(X_test)
    best_ann.evaluate(X_test_best, y_test, show_report=True)
    best_ann.plot_confusion_matrix(X_test_best, y_test)
    best_ann.plot_loss_curve()

    # ── 8. SAVE MODEL TO DISK ─────────────────────────────────────────
    print("\n[STEP 7] Saving model ...")
    save_model(best_model_data)

    print("\n[DONE] Training complete!")
    print("       Run 'python test.py' to test on new images.")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    train()
