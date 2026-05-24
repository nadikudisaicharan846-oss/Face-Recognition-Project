# =============================================================================
# ann.py — Artificial Neural Network for Face Classification
# =============================================================================
#
# WHAT IS AN ANN?
# ---------------
# An Artificial Neural Network (ANN) is a machine-learning model loosely
# inspired by the human brain.  It consists of:
#
#   Input Layer  → receives the K-dimensional PCA signature
#   Hidden Layer(s) → learn non-linear patterns via weighted connections
#   Output Layer → one neuron per person; highest activation = predicted class
#
# TRAINING (Backpropagation)
# --------------------------
#   1. Forward Pass  : input flows through the network → prediction
#   2. Compute Loss  : difference between prediction and true label
#   3. Backward Pass : gradients of the loss are propagated backwards
#   4. Weight Update : weights adjusted to reduce the loss (gradient descent)
#
# We use sklearn's MLPClassifier which implements all of the above internally.
# The network architecture used:
#
#   [K inputs] → [256 neurons] → [128 neurons] → [num_classes outputs]
#
# Activation : ReLU  (Rectified Linear Unit) — fast, works well for images
# Solver     : Adam  (Adaptive Moment Estimation) — modern gradient descent
# =============================================================================

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# ANN Classifier Class
# =============================================================================
class ANNClassifier:
    """
    Wraps sklearn's MLPClassifier with helper methods for:
        - Training
        - Prediction
        - Imposter detection (reject unknown faces)
        - Evaluation & visualisation
    """

    def __init__(self, hidden_layers=(256, 128), max_iter=500,
                 random_state=42, learning_rate=0.001):
        """
        Args:
            hidden_layers : tuple of neurons per hidden layer.
                            (256, 128) means two hidden layers.
            max_iter      : maximum training epochs.
            random_state  : seed for reproducibility.
            learning_rate : initial learning rate for Adam optimiser.
        """
        self.hidden_layers   = hidden_layers
        self.max_iter        = max_iter
        self.random_state    = random_state
        self.learning_rate   = learning_rate
        self.model           = None
        self.class_names     = None     # list of person names

        # Imposter detection threshold
        # If the ANN's maximum output probability < threshold → IMPOSTER
        self.imposter_threshold = 0.5

    # ------------------------------------------------------------------
    def build_model(self):
        """Creates (but does not train) the MLPClassifier."""
        self.model = MLPClassifier(
            hidden_layer_sizes = self.hidden_layers,
            activation         = 'relu',        # ReLU for hidden layers
            solver             = 'adam',         # Adam optimiser
            learning_rate_init = self.learning_rate,
            max_iter           = self.max_iter,
            random_state       = self.random_state,
            early_stopping     = True,           # stop if val accuracy stops improving
            validation_fraction= 0.1,            # 10% of training data for validation
            n_iter_no_change   = 20,             # patience
            verbose            = False
        )
        return self.model

    # ------------------------------------------------------------------
    def train(self, X_train, y_train, class_names):
        """
        Trains the ANN on PCA signatures.

        Args:
            X_train     : numpy array, shape (n_train, K) — PCA signatures
            y_train     : numpy array, shape (n_train,)  — integer labels
            class_names : list of person name strings
        """
        self.class_names = class_names

        print("\n[ANN] Building neural network ...")
        print(f"      Architecture: {X_train.shape[1]} → "
              f"{' → '.join(str(n) for n in self.hidden_layers)} → {len(class_names)}")
        print(f"      Training samples : {len(X_train)}")
        print(f"      Max iterations   : {self.max_iter}")

        self.build_model()

        print("[ANN] Training (this may take a moment) ...")
        self.model.fit(X_train, y_train)

        # Report training accuracy
        train_preds = self.model.predict(X_train)
        train_acc   = accuracy_score(y_train, train_preds) * 100
        print(f"[ANN] Training accuracy : {train_acc:.2f}%")
        print(f"[ANN] Iterations run    : {self.model.n_iter_}")

    # ------------------------------------------------------------------
    def predict(self, X):
        """
        Predicts class labels for PCA-projected face signatures.

        Args:
            X : numpy array, shape (n_samples, K)

        Returns:
            predictions : numpy array of integer labels, shape (n_samples,)
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return self.model.predict(X)

    # ------------------------------------------------------------------
    def predict_with_names(self, X):
        """
        Predicts and maps integer labels → person names.

        Returns:
            list of predicted names
        """
        preds = self.predict(X)
        return [self.class_names[p] for p in preds]

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """
        Returns class probability scores for each sample.

        Returns:
            proba : numpy array, shape (n_samples, num_classes)
        """
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    def predict_single(self, signature):
        """
        Predicts identity for ONE face signature, with imposter detection.

        The ANN outputs a probability for each class.  If the MAXIMUM
        probability is below self.imposter_threshold, the face is declared
        an IMPOSTER (not enrolled in the system).

        Args:
            signature : numpy array, shape (K,)

        Returns:
            name       : predicted person name OR "IMPOSTER"
            confidence : max probability (0.0 – 1.0)
        """
        sig_2d = signature.reshape(1, -1)               # sklearn needs 2D

        proba      = self.model.predict_proba(sig_2d)[0]  # shape (num_classes,)
        max_prob   = np.max(proba)
        pred_label = np.argmax(proba)

        if max_prob < self.imposter_threshold:
            return "IMPOSTER", max_prob
        else:
            return self.class_names[pred_label], max_prob

    # ------------------------------------------------------------------
    def evaluate(self, X_test, y_test, show_report=True):
        """
        Evaluates the model on a test set and prints performance metrics.

        Args:
            X_test      : PCA signatures, shape (n_test, K)
            y_test      : true integer labels, shape (n_test,)
            show_report : if True, prints a per-class precision/recall table

        Returns:
            accuracy : float (0–100)
        """
        preds    = self.predict(X_test)
        accuracy = accuracy_score(y_test, preds) * 100

        print(f"\n[ANN] Test Accuracy : {accuracy:.2f}%")

        if show_report and self.class_names:
            print("\n[ANN] Classification Report:")
            print(classification_report(
    y_test,
    preds,
    labels=sorted(set(y_test)),
    target_names=self.class_names[:len(set(y_test))],
    zero_division=0
))

        return accuracy

    # ------------------------------------------------------------------
    def plot_confusion_matrix(self, X_test, y_test):
        """
        Plots a heatmap confusion matrix for the test set.

        Rows = True labels, Columns = Predicted labels.
        Diagonal = correct predictions.
        """
        preds = self.predict(X_test)
        cm    = confusion_matrix(y_test, preds)

        plt.figure(figsize=(max(6, len(self.class_names)), max(5, len(self.class_names) - 1)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title("Confusion Matrix", fontsize=14)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150)
        plt.show()
        print("[INFO] Confusion matrix saved to 'confusion_matrix.png'")

    # ------------------------------------------------------------------
    def plot_loss_curve(self):
        """
        Plots the training loss curve (how loss decreased over iterations).

        A well-trained model should show a steadily decreasing loss curve.
        """
        if self.model is None or not hasattr(self.model, 'loss_curve_'):
            print("[WARNING] No loss curve available.")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(self.model.loss_curve_, color='tomato', linewidth=2)
        plt.title("ANN Training Loss Curve", fontsize=14)
        plt.xlabel("Iteration (Epoch)")
        plt.ylabel("Cross-Entropy Loss")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("loss_curve.png", dpi=150)
        plt.show()
        print("[INFO] Loss curve saved to 'loss_curve.png'")
