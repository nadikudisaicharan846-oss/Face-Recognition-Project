# 🧠 Face Recognition using PCA + ANN

> Implementation of Principal Component Analysis (PCA) combined with an
> Artificial Neural Network (ANN) for automated face recognition.

---

## 📋 Project Overview

This project builds a **face recognition system** using two powerful techniques:

| Component | Purpose |
|-----------|---------|
| **PCA (Eigenfaces)** | Reduces 10,000-pixel face images to a compact K-dimensional signature |
| **ANN (MLPClassifier)** | Learns to classify faces based on their PCA signatures |

The system can:
- Recognise enrolled persons with high accuracy
- Detect **imposters** (people not in the training set)
- Show how accuracy changes with different numbers of principal components (K)

---

## 📁 Folder Structure

```
FaceRecognition/
│
├── dataset/                ← Put face images here
│   ├── person1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── person2/
│   │   └── img1.jpg
│   └── test_images/        ← Optional separate test set
│       ├── person1/
│       └── imposters/
│
├── pca.py                  ← PCA (sklearn + manual implementation)
├── ann.py                  ← ANN classifier with imposter detection
├── train.py                ← Main training pipeline
├── test.py                 ← Testing / recognition pipeline
├── utils.py                ← Image I/O, visualisation, model saving
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## ⚙️ Installation

### Step 1 — Clone or Download the project

```bash
git clone <your-repo-url>
cd FaceRecognition
```

### Step 2 — Create a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### Step 3 — Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4 — Download the Dataset

Download from:
```
https://github.com/robaita/introduction_to_machine_learning/blob/main/dataset.zip
```
Extract and place each person's folder inside the `dataset/` directory.

---

## 🚀 How to Run

### Train the Model

```bash
python train.py
```

This will:
- Load all images
- Apply PCA for different K values
- Train ANN for each K
- Plot Accuracy vs K graph
- Save the best model to `saved_model.pkl`

### Test / Recognise a Face

```bash
# Batch test on test_images/ folder:
python test.py

# Test a single image file:
python test.py path/to/face.jpg

# Test an imposter:
python test.py path/to/unknown_person.jpg
```

---

## 📚 Required Libraries

| Library | Use |
|---------|-----|
| `numpy` | Matrix operations, eigenvalue decomposition |
| `opencv-python` | Image reading, display, preprocessing |
| `scikit-learn` | PCA, MLPClassifier (ANN), evaluation metrics |
| `matplotlib` | Eigenface & accuracy-vs-K visualisation |
| `seaborn` | Confusion matrix heatmap |
| `scipy` | Optional advanced linear algebra |

---

## 🧮 PCA Explanation (Eigenfaces Method)

**Principal Component Analysis (PCA)** finds the directions in which data varies the most.

### Steps:

1. **Build face matrix A** (mn × p): Each face image (m×n pixels) is flattened into a column vector. p = number of images.

2. **Compute mean face M** (mn × 1): Average of all face vectors.

3. **Mean subtraction Φ = A - M**: Remove the average to focus on how faces differ.

4. **Surrogate Covariance C = ΦᵀΦ** (p × p): The real covariance (mn×mn) is too large. This trick gives the same non-zero eigenvalues.

5. **Eigendecomposition**: Find eigenvectors V and eigenvalues λ of C.

6. **Map back to pixel space**: Real eigenfaces U = ΦV (mn × p).

7. **Select top-K eigenfaces**: Highest eigenvalues = most face variation captured.

8. **Generate signatures**: Project each face onto eigenfaces → compact K-dim vector.

> 💡 **Key Insight**: Instead of comparing 10,000-pixel images directly, we compare K-dimensional signatures (e.g., K=50). This is much faster and more accurate!

---

## 🤖 ANN Explanation

**Artificial Neural Network (ANN)** learns to classify PCA signatures into person identities.

### Architecture:
```
[K inputs] → [256 neurons, ReLU] → [128 neurons, ReLU] → [N outputs, Softmax]
```

### Training (Backpropagation):
1. **Forward pass**: Input flows through layers → produces prediction
2. **Loss computation**: Cross-entropy between prediction and true label
3. **Backward pass**: Gradient of loss propagated back through layers
4. **Weight update**: Adam optimiser adjusts weights to reduce loss
5. **Repeat** until convergence

### Imposter Detection:
- The final layer outputs a probability for each enrolled person
- If `max(probabilities) < threshold (0.5)` → face declared **IMPOSTER**
- Threshold can be tuned: lower = stricter, higher = more permissive

---

## 📊 Sample Outputs

After training, the following files are generated:

| File | Description |
|------|-------------|
| `eigenfaces.png` | Top-10 eigenfaces visualised |
| `accuracy_vs_k.png` | Line graph: K vs accuracy |
| `confusion_matrix.png` | Heatmap: true vs predicted labels |
| `loss_curve.png` | ANN training loss over iterations |
| `saved_model.pkl` | Saved trained model |

---

## 🔮 Future Improvements

1. **Deep Learning**: Replace ANN with a CNN (Convolutional Neural Network) for better accuracy on larger datasets.
2. **Real-time Recognition**: Use webcam feed with OpenCV for live face recognition.
3. **LDA (Fisher Faces)**: Use Linear Discriminant Analysis instead of PCA for better class separation.
4. **Data Augmentation**: Add rotated/flipped versions of images to improve robustness.
5. **Face Detection**: Add face detection (Haar cascades / MTCNN) before recognition so the system works on natural photos.
6. **GUI**: Build a Tkinter or web interface for easier use.

---

## 👩‍💻 Author Notes

- The `SklearnPCA` class is used by default (fast and reliable)
- To use the manual PCA implementation, change the import in `train.py`:
  ```python
  from pca import ManualPCA as SklearnPCA
  ```
- All mathematical steps are commented inline in `pca.py`
