# =============================================================================
# utils.py — Utility Functions for Face Recognition System
# =============================================================================
# This file handles:
#   - Loading face images from folder-based dataset
#   - Preprocessing (grayscale, resize, flatten)
#   - Saving & loading trained models
#   - Displaying results using OpenCV
# =============================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle


# ---------------------------------------------------------------------------
# CONSTANTS — change these to match your dataset
# ---------------------------------------------------------------------------
IMAGE_SIZE = (100, 100)          # All images will be resized to 100x100 pixels
DATASET_PATH = "dataset"         # Root folder containing sub-folders per person
MODEL_PATH = "saved_model.pkl"   # Where the trained ANN model will be saved


# ---------------------------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------------------------
def load_dataset(dataset_path=DATASET_PATH, image_size=IMAGE_SIZE):
    """
    Reads all face images from a folder-based dataset.

    Expected folder structure:
        dataset/
            person1/
                img1.jpg
                img2.jpg
            person2/
                img1.jpg
            ...

    Returns:
        images  : numpy array of shape (num_images, height*width)
                  Each row = one flattened grayscale image
        labels  : numpy array of integer class labels (0, 1, 2, ...)
        names   : list of person names (folder names)
    """
    images = []
    labels = []
    names  = []

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset folder '{dataset_path}' not found!\n"
            "Please download the dataset and place it in the project folder."
        )

    # Each sub-folder = one person
    class_id = 0
    for person_name in sorted(os.listdir(dataset_path)):
        person_folder = os.path.join(dataset_path, person_name)

        # Skip files; only process sub-folders
        if not os.path.isdir(person_folder):
            continue

        names.append(person_name)

        for img_file in sorted(os.listdir(person_folder)):
            img_path = os.path.join(person_folder, img_file)

            # Read image using OpenCV
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [WARNING] Could not read: {img_path} — skipping.")
                continue

            # Step 1: Convert to Grayscale
            # Face structure is captured well in grayscale; color adds noise
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Step 2: Resize to fixed size (100x100)
            # All images MUST be the same size for PCA to work
            resized = cv2.resize(gray, image_size)

            # Step 3: Flatten 2D image → 1D vector
            # A 100x100 image becomes a 10000-dimensional vector
            flat = resized.flatten().astype(np.float64)

            images.append(flat)
            labels.append(class_id)

        class_id += 1

    if len(images) == 0:
        raise ValueError(
            "No images found in dataset! "
            "Make sure the dataset folder contains sub-folders with .jpg/.png images."
        )

    images = np.array(images)   # Shape: (num_images, 10000)
    labels = np.array(labels)   # Shape: (num_images,)

    print(f"[INFO] Dataset loaded: {len(images)} images | {len(names)} persons")
    return images, labels, names


# ---------------------------------------------------------------------------
# 2. VISUALIZE EIGENFACES
# ---------------------------------------------------------------------------
def show_eigenfaces(eigenfaces, image_size=IMAGE_SIZE, num_faces=10):
    """
    Displays the top eigenfaces (principal components) as images.

    Eigenfaces look like ghostly face images — they capture the
    directions of maximum variance across all training faces.

    Args:
        eigenfaces : numpy array, shape (k, image_size[0]*image_size[1])
        image_size : tuple (height, width)
        num_faces  : how many eigenfaces to display
    """
    n = min(num_faces, len(eigenfaces))
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 3))
    fig.suptitle("Top Eigenfaces (Principal Components)", fontsize=14)

    for i in range(n):
        # Reshape the 1D eigenvector back into a 2D image
        face_img = eigenfaces[i].reshape(image_size)

        # Normalize pixel values to 0-255 for display
        face_img = cv2.normalize(face_img, None, 0, 255, cv2.NORM_MINMAX)

        axes[i].imshow(face_img, cmap='gray')
        axes[i].set_title(f"EF {i+1}", fontsize=9)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("eigenfaces.png", dpi=150)
    plt.show()
    print("[INFO] Eigenfaces saved to 'eigenfaces.png'")


# ---------------------------------------------------------------------------
# 3. PLOT ACCURACY VS K GRAPH
# ---------------------------------------------------------------------------
def plot_accuracy_vs_k(k_values, accuracies):
    """
    Plots a line graph showing how accuracy changes with different K values.

    K = number of principal components (eigenfaces) selected.
    Higher K → more information retained → usually better accuracy (up to a point).

    Args:
        k_values   : list of K values tested
        accuracies : list of corresponding accuracy percentages
    """
    plt.figure(figsize=(9, 5))
    plt.plot(k_values, accuracies, marker='o', color='royalblue',
             linewidth=2, markersize=8, label='Test Accuracy')

    # Highlight the best K
    best_idx = np.argmax(accuracies)
    plt.scatter(k_values[best_idx], accuracies[best_idx],
                color='red', zorder=5, s=120, label=f'Best K={k_values[best_idx]}')

    plt.title("Accuracy vs Number of Principal Components (K)", fontsize=14)
    plt.xlabel("K (Number of Eigenfaces / Principal Components)", fontsize=12)
    plt.ylabel("Classification Accuracy (%)", fontsize=12)
    plt.xticks(k_values)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_vs_k.png", dpi=150)
    plt.show()
    print("[INFO] Accuracy-vs-K graph saved to 'accuracy_vs_k.png'")


# ---------------------------------------------------------------------------
# 4. SAVE MODEL
# ---------------------------------------------------------------------------
def save_model(model_data, path=MODEL_PATH):
    """
    Saves the trained model and PCA data to disk using pickle.

    model_data is a dictionary containing:
        - 'ann'        : trained MLPClassifier
        - 'mean_face'  : mean face vector
        - 'eigenfaces' : selected eigenfaces matrix
        - 'names'      : list of person names
        - 'k'          : number of components used
    """
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"[INFO] Model saved to '{path}'")


# ---------------------------------------------------------------------------
# 5. LOAD MODEL
# ---------------------------------------------------------------------------
def load_model(path=MODEL_PATH):
    """
    Loads a previously saved model from disk.

    Returns the model_data dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved model found at '{path}'. "
            "Please run train.py first."
        )
    with open(path, 'rb') as f:
        model_data = pickle.load(f)
    print(f"[INFO] Model loaded from '{path}'")
    return model_data


# ---------------------------------------------------------------------------
# 6. DISPLAY RECOGNITION RESULT (OpenCV window)
# ---------------------------------------------------------------------------
def display_result(image_path, predicted_name, confidence=None, image_size=IMAGE_SIZE):
    """
    Opens a window displaying the test face with the predicted identity.

    Args:
        image_path     : path to the test image file
        predicted_name : recognized person's name (or "IMPOSTER")
        confidence     : optional confidence/distance score
        image_size     : resize target
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARNING] Cannot display image: {image_path}")
        return

    # Resize for display
    display_img = cv2.resize(img, (300, 300))

    # Choose label color: green for known, red for imposter
    color = (0, 255, 0) if predicted_name != "IMPOSTER" else (0, 0, 255)

    label = f"ID: {predicted_name}"
    if confidence is not None:
        label += f"  (dist={confidence:.1f})"

    # Draw filled rectangle behind text for readability
    cv2.rectangle(display_img, (0, 260), (300, 300), (0, 0, 0), -1)
    cv2.putText(display_img, label, (5, 285),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition Result", display_img)
    print(f"[RESULT] Predicted: {predicted_name}  |  Press any key to close window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# 7. PREPROCESS A SINGLE IMAGE (for testing)
# ---------------------------------------------------------------------------
def preprocess_single_image(image_path, image_size=IMAGE_SIZE):
    """
    Reads, converts to grayscale, resizes, and flattens ONE image.

    Used during the testing/recognition phase.

    Returns:
        flat_vector : numpy array of shape (image_size[0]*image_size[1],)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, image_size)
    flat    = resized.flatten().astype(np.float64)
    return flat
