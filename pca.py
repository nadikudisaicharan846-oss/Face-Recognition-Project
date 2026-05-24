# =============================================================================
# pca.py — PCA (Principal Component Analysis) for Face Recognition
# =============================================================================
#
# WHAT IS PCA?
# ------------
# PCA finds the directions (called principal components or eigenfaces) in which
# the face data varies the most.  By projecting faces onto these directions we
# get a compact, noise-reduced representation called the "signature" of each face.
#
# MATHEMATICAL STEPS  (Turk & Pentland, 1991 — "Eigenfaces")
# -----------------------------------------------------------
#   1.  Stack all training face vectors into a matrix  A  (mn × p)
#   2.  Compute mean face  M  (mn × 1)
#   3.  Subtract mean from every face  →  Φ  (mean-zero matrix)
#   4.  Compute surrogate covariance  C = Φᵀ Φ  (p × p)  ← trick!
#   5.  Find eigenvectors V and eigenvalues λ of C
#   6.  Project back:  real eigenfaces  U = Φ · V  (mn × p)
#   7.  Pick top-K eigenfaces (highest eigenvalues)
#   8.  Project every training face onto those K eigenfaces  → signatures
#
# TWO IMPLEMENTATIONS PROVIDED:
#   - SklearnPCA  :  uses sklearn.decomposition.PCA  (fast, recommended)
#   - ManualPCA   :  pure NumPy implementation (educational, matches the maths above)
# =============================================================================

import numpy as np
from sklearn.decomposition import PCA as SklearnPCALib
from sklearn.preprocessing import StandardScaler


# =============================================================================
# OPTION A — Sklearn-based PCA  (recommended for coursework)
# =============================================================================
class SklearnPCA:
    """
    Wrapper around sklearn's PCA.

    Attributes after fit():
        pca          : fitted sklearn PCA object
        eigenfaces   : top-K principal components, shape (K, mn)
        mean_face    : mean face vector, shape (mn,)
        k            : number of components selected
    """

    def __init__(self, k=50):
        """
        Args:
            k : number of principal components (eigenfaces) to keep.
                Typical range: 10 – 150.
        """
        self.k = k
        self.pca = None
        self.eigenfaces  = None
        self.mean_face   = None

    # ------------------------------------------------------------------
    def fit(self, X):
        """
        Trains PCA on the training face matrix X.

        Args:
            X : numpy array, shape (num_train_images, mn)
                Each row is one flattened face image.
        """
        print(f"\n[PCA] Fitting sklearn PCA with K={self.k} components ...")

        # Step 1 — Compute & store mean face (done internally by sklearn too,
        #          but we store it separately so we can subtract it at test time)
        self.mean_face = np.mean(X, axis=0)          # shape (mn,)

        # Step 2 — Fit PCA: finds top-K directions of maximum variance
        self.pca = SklearnPCALib(n_components=self.k, whiten=False)
        self.pca.fit(X)

        # Step 3 — Store eigenfaces: each row is one principal component
        self.eigenfaces = self.pca.components_       # shape (K, mn)

        variance_explained = np.sum(self.pca.explained_variance_ratio_) * 100
        print(f"[PCA] Done. Variance explained by top {self.k} components: "
              f"{variance_explained:.1f}%")

    # ------------------------------------------------------------------
    def transform(self, X):
        """
        Projects face images onto the eigenface space.

        Steps:
            1. Subtract mean face  (mean-zero)
            2. Dot-product with eigenfaces  →  K-dimensional signature

        Args:
            X : numpy array, shape (n_samples, mn)

        Returns:
            signatures : numpy array, shape (n_samples, K)
        """
        # Mean subtraction (centering)
        X_centered = X - self.mean_face               # broadcast subtraction

        # Project onto K eigenfaces
        # signature_i = Φ_i · EigenFaces^T
        signatures = self.pca.transform(X)            # sklearn handles centering
        return signatures                              # shape (n_samples, K)

    # ------------------------------------------------------------------
    def fit_transform(self, X):
        """Convenience: fit then transform in one call."""
        self.fit(X)
        return self.transform(X)

    # ------------------------------------------------------------------
    def transform_single(self, x):
        """
        Projects a single test face vector.

        Args:
            x : numpy array, shape (mn,)  — one flattened face
        Returns:
            signature : numpy array, shape (K,)
        """
        x_2d = x.reshape(1, -1)             # sklearn needs 2D input
        return self.pca.transform(x_2d)[0]  # return 1D


# =============================================================================
# OPTION B — Manual PCA  (educational; matches the PDF maths exactly)
# =============================================================================
class ManualPCA:
    """
    Pure NumPy PCA following the Turk & Pentland Eigenfaces method.

    Attributes after fit():
        mean_face    : mean face vector, shape (mn,)
        eigenfaces   : selected eigenfaces, shape (K, mn)
        eigenvalues  : top-K eigenvalues (descending order)
        k            : number of components selected
    """

    def __init__(self, k=50):
        self.k = k
        self.mean_face  = None
        self.eigenfaces = None
        self.eigenvalues = None

    # ------------------------------------------------------------------
    def fit(self, X):
        """
        Fits Manual PCA step by step.

        Args:
            X : numpy array, shape (p, mn)   — p training images, each mn pixels
        """
        print(f"\n[ManualPCA] Fitting with K={self.k} components ...")

        p, mn = X.shape          # p = number of images, mn = image pixels

        # ── Step 1: Mean Face ──────────────────────────────────────────
        # M = (1/p) Σ Face_i     → shape (mn,)
        self.mean_face = np.mean(X, axis=0)

        # ── Step 2: Mean-Zero (subtract mean from every face) ─────────
        # Φ = Face_Db - M        → shape (p, mn)
        Phi = X - self.mean_face          # broadcasting handles (p, mn) - (mn,)

        # ── Step 3: Surrogate Covariance Matrix ───────────────────────
        # Instead of C = Φ·Φᵀ  (mn×mn — too large!)
        # We compute  C_surrogate = Φᵀ·Φ  (p×p — manageable!)
        # The non-zero eigenvectors of the real covariance map 1-to-1
        # with those of the surrogate covariance.
        C = (Phi @ Phi.T) / (p - 1)      # shape (p, p)

        print(f"[ManualPCA] Surrogate covariance matrix shape: {C.shape}")

        # ── Step 4: Eigendecomposition of surrogate covariance ────────
        # eigenvalues  λ  shape (p,)
        # eigenvectors V  shape (p, p)  — columns are eigenvectors
        eigenvalues, V = np.linalg.eigh(C)

        # np.linalg.eigh returns eigenvalues in ASCENDING order → reverse
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        V = V[:, idx]                     # reorder columns accordingly

        # ── Step 5: Project back to mn-dimensional space ──────────────
        # Real eigenfaces U = Φᵀ · V   → shape (mn, p)
        # Each COLUMN of U is one eigenface (principal component in pixel space)
        U = Phi.T @ V                     # shape (mn, p)

        # ── Step 6: Normalise each eigenface to unit length ───────────
        norms = np.linalg.norm(U, axis=0, keepdims=True)
        norms[norms == 0] = 1             # avoid division by zero
        U = U / norms                     # shape (mn, p)

        # ── Step 7: Select top-K eigenfaces ───────────────────────────
        K = min(self.k, p)
        self.eigenfaces  = U[:, :K].T    # shape (K, mn) — rows are eigenfaces
        self.eigenvalues = eigenvalues[:K]

        variance_total    = np.sum(np.abs(eigenvalues))
        variance_captured = np.sum(np.abs(self.eigenvalues))
        pct = (variance_captured / variance_total * 100) if variance_total > 0 else 0
        print(f"[ManualPCA] Done. Variance captured by top {K} components: {pct:.1f}%")

    # ------------------------------------------------------------------
    def transform(self, X):
        """
        Projects face images onto the K eigenfaces.

        Signature_i = EigenFaces (K×mn) · Φ_i (mn×1)

        Args:
            X : numpy array, shape (n_samples, mn)

        Returns:
            signatures : numpy array, shape (n_samples, K)
        """
        # Mean-zero
        Phi = X - self.mean_face          # (n_samples, mn)

        # Project: each row of Phi dot-producted with each eigenface row
        # Result shape: (n_samples, K)
        signatures = Phi @ self.eigenfaces.T
        return signatures

    # ------------------------------------------------------------------
    def fit_transform(self, X):
        """Convenience: fit then transform in one call."""
        self.fit(X)
        return self.transform(X)

    # ------------------------------------------------------------------
    def transform_single(self, x):
        """
        Projects a single test face vector.

        Args:
            x : numpy array, shape (mn,)
        Returns:
            signature : numpy array, shape (K,)
        """
        phi = x - self.mean_face          # mean subtraction
        return self.eigenfaces @ phi      # shape (K,)
