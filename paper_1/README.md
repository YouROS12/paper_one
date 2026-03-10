# PlantWild Disease Detection Project

This project analyzes plant diseases using two main approaches:
1.  **Feature Extraction + Classical ML**: Extracting features using a pre-trained YOLOv8 model, reducing dimensionality with PCA, and training classifiers like SVM (cuSVM), XGBoost, and Random Forest.
2.  **Deep Learning Fine-Tuning**: Fine-tuning an EfficientNet-B0 model directly on the image data.

## Project Structure

The codebase has been restructured into two main Jupyter Notebooks for clarity and reproducibility:

### 1. `src/preprocessing.py`
**Goal**: Prepare the data for training.
**Key Steps**:
*   **Feature Extraction**: Loads the YOLOv8 model and extracts features from images in the dataset.
*   **Dimensionality Reduction**: Applies Incremental PCA to the extracted features to reduce the feature space size while retaining variance.
*   **Encoding**: Encodes class labels (e.g., mapping specific diseases to broader categories like "fungal_rust").
*   **Output**: Saves the processed features and labels as `.npy` files (e.g., `x_train_pool_pca.npy`, `y_mlc_train_pool_encoded.npy`) and pickle files.

### 2. `src/training_and_evaluation.py`
**Goal**: Train models and evaluate their performance.
**Key Steps**:
*   **Classical ML Training**: Loads the processed `.npy` data and trains various classifiers (SVM, XGBoost, RandomForest, MLP).
    *   Includes Hyperparameter tuning using `GridSearchCV`.
    *   Utilizes GPU acceleration (cuSVM) where available.
*   **Deep Learning Training**: Contains the PyTorch training loop for fine-tuning EfficientNet-B0.
*   **Evaluation**: Generates detailed classification reports, confusion matrices, and reliability diagrams.
*   **Analysis**: Produces LaTeX tables for reporting results.

## Getting Started

1.  **Run `src/preprocessing.py`**:
    *   Ensure your dataset paths are correctly set in the "Configuration" section.
    *   Run this notebook to generate the feature files. This step only needs to be done once or when the dataset changes.

2.  **Run `src/training_and_evaluation.py`**:
    *   This notebook will load the features generated in the previous step.
    *   You can choose to run specific experiments (e.g., just SVM or just the Deep Learning Loop).

## Experiments Guide

### How to Run the Experiments
The project uses a configuration-based approach. specific variables in the scripts control which experiment is active.

#### Experiment 1: Backbone Benchmark
1.  Open `src/preprocessing.py`.
2.  Change `CONFIG['BACKBONE']` to `'resnet50'` or `'efficientnet_b0'`.
3.  Run the script to generate new feature files (e.g., `x_train_resnet50_ipca.npy`).
4.  Open `src/training_and_evaluation.py`.
5.  Change `DATA_CONFIG['BACKBONE']` to match.
6.  Run the training cell to see how the new backbone performs.

#### Experiment 2: Compression Benchmark
1.  Open `src/preprocessing.py`.
2.  Change `CONFIG['COMPRESSION']` to `'svd'` (TruncatedSVD).
3.  Run the script.
4.  Update `src/training_and_evaluation.py` config and compare results.

#### Experiment 3: Label Engineering
1.  Open `src/training_and_evaluation.py`.
2.  Change `DATA_CONFIG['LABEL_SET']` to `'11_classes'` (Treatment-based) or `'19_classes'` (Visual-based).
3.  Run the training cell. Note how accuracy likely improves with 11 classes due to simpler boundaries.

#### Experiment 4: Speed Benchmark
1.  Open `src/training_and_evaluation.py`.
2.  Run **Cell 5**.
3.  It will automatically load the architecture defined in your config and measure mean inference time on the CPU.

## Requirements
*   Python 3.x
*   PyTorch, torchvision
*   Ultralytics (YOLOv8)
*   scikit-learn
*   cuML (for GPU-accelerated SVM, optional but recommended)
*   pandas, numpy, matplotlib, seaborn
*   timm (for EfficientNet)
