
You are acting as the **Lead Academic Editor** for a research paper on **AI in Plant Pathology**. Your task is to synchronize the LaTeX manuscript (`paper_updated/paper_exported_from_overleaf/main.tex`) with the final version of the experimental codebase (`src/`).

**Input Files to Read:**
1.  `src/preprocessing.py`: Defines the data processing pipeline, feature extraction (YOLO layer 8), and label engineering (11-class mapping).
2.  `src/training_and_evaluation.py`: Defines the classifier training (SVC settings), evaluation metrics, and experimental loop.
3.  `src/plot_results.py`: Generates the figures (confusion matrices, bar charts). Use this to verify figure captions and descriptions.
4.  `paper_updated/paper_exported_from_overleaf/main.tex`: The LaTeX source file that needs to be updated.

**Your Goal:**
Ensure that every claim in the **Methodology**, **Experimental Design**, and **Results** sections of `main.tex` is strictly supported by the code in `src/`. If the code differs from the text, UPDATE THE TEXT to match the code.

**Key Areas to Verify & Update (CRITICAL):**

### 1. Methodology Consistency
*   **Feature Extraction**:
    *   **YOLOv8**: The code uses `YOLO_FEATURE_LAYER = 8` (a C2f block). Verify that Section \ref{sec:methodology} explicitly mentions this specific layer choice.
    *   **Backbones**: Confirm the paper lists **ResNet50**, **EfficientNet-B0**, and **YOLOv8m**.
*   **Dimensionality Reduction**:
    *   The code uses **Incremental PCA (IPCA)** with `batch_size=512` and **Truncated SVD**, reducing dimensions to **n=100**. Ensure the paper reflects this exact configuration, emphasizing the memory efficiency of IPCA.
*   **Label Engineering (The "11-Class" Strategy)**:
    *   Locate the `mapping_11_classes` dictionary in `src/preprocessing.py`.
    *   **Action**: Update the "Treatment-Based Grouping" description in Section \ref{sec:methodology} to list the *exact* super-classes defined in the code (e.g., `fungal_rust`, `fungal_powdery_mildew`, `abiotic_disorder`, etc.). The paper uses vague terms; make them precise code-matched terms where appropriate.

### 2. Experimental Setup
*   **Classifier**:
    *   Check `src/training_and_evaluation.py`. The classifier is an **SVC** with `kernel='rbf'`, `class_weight='balanced'`, and `probability=True`. Ensure Section \ref{sec:methodology} mentions these specific hyperparameters.
*   **Reproducibility**:
    *   The code now saves trained models as `.joblib` files. Mention this in the implementation details if relevant.

### 3. Results (Table Updates)
*   **Table \ref{tab:master_results_summary}**: This table lists performance for different backbones.
    *   *If you can see the results in the code comments or output files*: Update the numbers in this LaTeX table to match the final run.
    *   *If exact numbers are missing*: Ensure the *structure* of the table matches the definition of experiments in `src/preprocessing.py` (i.e., ResNet, EfficientNet, YOLO vs IPCA, SVD).
    *   **Champion Model**: The paper claims **87.52%** accuracy for YOLOv8m + IPCA. VALIDATE this against the strongest result implied by the codebase or available CSVs.

### 4. Tone and Formatting
*   **Output Format**: Provide the **full updated LaTeX code** for the modified sections, or the entire file if requested.
*   **Precision**: Use specific terms (e.g., "Layer 8 C2f block" instead of "an intermediate layer").
*   **Style**: Maintain high-quality academic English.

**Immediate Action:**
Read the `src/` files first to establish the "Ground Truth". Then, edit `main.tex` to align perfectly with that truth.
