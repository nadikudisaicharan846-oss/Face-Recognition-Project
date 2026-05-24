# 🎓 Viva Questions & Answers
## PCA + ANN Face Recognition System

---

### SECTION 1 — PCA (Principal Component Analysis)

---

**Q1. What is PCA and why do we use it in face recognition?**

**A:** PCA (Principal Component Analysis) is a dimensionality reduction technique. In face recognition, each image is a high-dimensional vector (e.g., a 100×100 image = 10,000 dimensions). Comparing images directly in this high-dimensional space is computationally expensive and affected by noise. PCA finds the directions (principal components) of maximum variance in the data and projects the images onto these directions, reducing them to a much smaller K-dimensional vector called a "signature" or "feature vector." This makes recognition faster, more accurate, and more noise-resistant.

---

**Q2. What is an Eigenface?**

**A:** Eigenfaces are the principal components of the face dataset visualised as images. Each eigenface is a direction in the face space that captures a particular pattern of variation across all training faces. The top eigenfaces capture the most significant variations (e.g., lighting differences, expression changes, head orientation). Any face in the dataset can be approximately reconstructed as a weighted sum of eigenfaces.

---

**Q3. Why do we subtract the mean face?**

**A:** Subtracting the mean face (also called "mean centering" or "mean zero") removes the average illumination and overall face structure from each image. PCA needs centered data to correctly identify the directions of maximum *variation*, not maximum *magnitude*. Without mean subtraction, the first principal component would simply point in the direction of the mean face itself, not in the direction of maximum variation.

---

**Q4. What is the Surrogate Covariance Matrix and why do we use it?**

**A:** The true covariance matrix of face data would be (mn × mn) in size — for 100×100 images that is 10,000 × 10,000 = 100 million entries. This is too large to store or compute the eigendecomposition of. Turk & Pentland (1991) showed that we can compute the surrogate covariance matrix C = ΦᵀΦ of size (p × p) instead, where p is the number of training images. This matrix has the same non-zero eigenvalues as the full covariance matrix, and the real eigenfaces can be recovered by projecting back: U = Φ·V.

---

**Q5. What does K represent? How do you choose the best K?**

**A:** K is the number of principal components (eigenfaces) selected after sorting by eigenvalue magnitude (descending). A higher K retains more information but also more noise. The optimal K is typically chosen by:
1. Plotting accuracy vs K and selecting the K at which accuracy peaks.
2. Using the "explained variance" criterion: choose K such that the selected components explain ≥ 95% of the total variance.
3. Cross-validation on a validation set.

In our experiment, we test K ∈ {5, 10, 20, 30, 50, 75, 100} and pick the best.

---

**Q6. What is the difference between PCA and LDA for face recognition?**

**A:**
| | PCA | LDA |
|---|---|---|
| Goal | Maximize overall variance | Maximize between-class / minimize within-class variance |
| Labels needed | No (unsupervised) | Yes (supervised) |
| Components | ≤ p-1 | ≤ num_classes-1 |
| Also known as | Eigenfaces | Fisherfaces |

LDA generally gives better recognition accuracy since it optimises for class separation, but PCA is simpler and works well as a first step.

---

**Q7. How is a test image recognised?**

**A:** Steps:
1. Read and preprocess the test image (grayscale, resize, flatten).
2. Subtract the training mean face (mean-zero).
3. Project onto the K eigenfaces → get a K-dimensional test signature.
4. Feed the signature to the trained ANN classifier.
5. The ANN outputs a probability for each enrolled person. The class with the highest probability is the predicted identity (if above the threshold).

---

### SECTION 2 — ANN (Artificial Neural Network)

---

**Q8. What is an Artificial Neural Network?**

**A:** An ANN is a computational model inspired by the structure of biological neural networks in the brain. It consists of layers of interconnected "neurons" (nodes). Each connection has a weight. The network learns the optimal weights through training using the backpropagation algorithm. In our system, the ANN takes the K-dimensional PCA signature as input and outputs the identity of the face.

---

**Q9. Explain the backpropagation algorithm.**

**A:** Backpropagation is the algorithm used to train ANNs:
1. **Forward pass**: Input data flows through the network layer by layer, producing a prediction.
2. **Loss calculation**: A loss function (cross-entropy for classification) measures the error between the prediction and true label.
3. **Backward pass**: The gradient (partial derivative) of the loss with respect to each weight is calculated using the chain rule of calculus, propagating backwards from output to input.
4. **Weight update**: Each weight is updated in the direction opposite to its gradient (gradient descent): `w = w - η·(∂L/∂w)` where η is the learning rate.
5. This process repeats for many iterations (epochs) until the loss converges.

---

**Q10. What activation function is used and why?**

**A:** We use **ReLU (Rectified Linear Unit)**: `f(x) = max(0, x)`.
- Advantages: computationally cheap, avoids the vanishing gradient problem that affects sigmoid/tanh, allows the network to learn sparse representations.
- It returns 0 for negative inputs (neuron "off") and x for positive inputs (neuron "on"), introducing the non-linearity needed to learn complex patterns.

---

**Q11. What is the Adam optimiser?**

**A:** Adam (Adaptive Moment Estimation) is an advanced gradient descent optimiser that:
- Maintains an exponentially decaying average of past gradients (momentum term).
- Maintains an exponentially decaying average of past squared gradients (adapts the learning rate per parameter).
- Combines the benefits of RMSProp and Momentum optimisers.
- Generally converges faster and requires less hyperparameter tuning than plain SGD.

---

**Q12. How does imposter detection work?**

**A:** When an unknown person (imposter) is presented, the ANN still outputs probabilities for all enrolled classes — but none of them will be very high. We detect imposters by checking: if `max(output probabilities) < threshold` (e.g., 0.50), the person is declared an IMPOSTER. The threshold is a hyperparameter: lower = stricter (fewer false accepts but more false rejects), higher = more permissive (more false accepts but fewer false rejects).

---

### SECTION 3 — System Design

---

**Q13. Why split data as 60% training and 40% testing?**

**A:** The split ensures the model is evaluated on data it has never seen during training, giving an honest estimate of generalisation performance. 60/40 is specified in the project requirements. In practice, common splits are 80/20 or using k-fold cross-validation for small datasets.

---

**Q14. What preprocessing steps are applied to images and why?**

**A:**
| Step | Why |
|------|-----|
| Grayscale conversion | Colour adds noise; face structure is captured by intensity |
| Resize to 100×100 | PCA requires all vectors to be the same size |
| Flatten to 1D | PCA and ANN work on 1D vectors, not 2D images |
| Float64 cast | Required for accurate numerical computation |

---

**Q15. What are the limitations of this system?**

**A:**
1. Sensitive to illumination, pose, and expression changes.
2. PCA is a linear method — cannot capture non-linear face variation well.
3. Performance degrades with very few training images per person.
4. Imposter detection relies on a threshold that must be manually tuned.
5. No face detection step — the input image must already be aligned and cropped to the face region.

---

**Q16. What is the complexity of the training phase?**

**A:**
- PCA fitting: O(p² · mn) for surrogate covariance + O(p³) for eigendecomposition (p = training images, mn = pixels).
- ANN training: O(epochs × n × L) where L is network layers and n is training samples.
- For a dataset with 400 images of size 100×100: PCA is very fast; ANN training takes a few seconds on a modern CPU.
