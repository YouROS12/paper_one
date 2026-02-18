
You are acting as the **Lead Academic Editor** for a research paper on **AI in Plant Pathology**. Your task is to synchronize the LaTeX manuscript (`paper_updated/paper_exported_from_overleaf/main.tex`) with the final version of the experimental codebase (`src/`).

**Input Files to Read:**
1.  `src/preprocessing.py`: Defines the data processing pipeline, feature extraction (YOLO layer 8), and label engineering (11-class mapping).
2.  `src/training_and_evaluation.py`: Defines the classifier training (SVC settings), evaluation metrics, and experimental loop.
3.  `src/plot_results.py`: Generates the figures (confusion matrices, bar charts). Use this to verify figure captions and descriptions.
4.  `archive/final_exp_plantwild_cusvm.py`: **Context for Model Choice**. This shows our preliminary exploration. We settled on SVC to demonstrate that **YOLOv8 features are so robust** that even a standard classifier yields high performance. The focus is on **Backbone Quality**, not Classifier Superiority.
5.  `paper_updated/paper_exported_from_overleaf/main.tex`: The LaTeX source file that needs to be updated.

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
*   **Classifier Rationale**:
    *   The paper's core argument is that **YOLOv8 features are so rich** that a standard classifier like **SVC** is sufficient to achieve high accuracy.
    *   **Action**: Ensure the narrative focuses on the **Efficiency** and **Feature Quality** of the YOLO backbone. Use `archive/final_exp_plantwild_cusvm.py` only to show that we explored options, but the final choice of SVC highlights the power of the features (i.e., "We don't need complex ensembles when the features are this good").
*   **Classifier Hyperparameters**:
    *   Check `src/training_and_evaluation.py`. The classifier is an **SVC** with `kernel='rbf'`, `class_weight='balanced'`, and `probability=True`. Ensure Section \ref{sec:methodology} mentions these specific hyperparameters.
*   **Reproducibility**:
    *   The code now saves trained models as `.joblib` files. Mention this in the implementation details if relevant.

### 3. Results (Table Updates)
*   **Table \ref{tab:master_results_summary}**: This table lists performance for different backbones.
    *   *If you can see the results in the code comments or output files*: Update the numbers in this LaTeX table to match the final run.
    *   *If exact numbers are missing*: Ensure the *structure* of the table matches the definition of experiments in `src/preprocessing.py` (i.e., ResNet, EfficientNet, YOLO vs IPCA, SVD).
    *   **Champion Model**: The paper claims **87.52%** accuracy for YOLOv8m + IPCA. VALIDATE this against the strongest result implied by the codebase or available CSVs.

### 4. Special Reviewer Instructions (MANDATORY UPDATES)
You must update the text to address these specific reviewer criticisms:

*   **Figure 1 Size**: "Figure 1 appears too small." -> **Action**: Update the LaTeX for `fig:system_architecture` to use `width=\textwidth` or `height=0.95\textheight` to maximize size.
*   **Dataset Bias Justification**: The reviewer asks why PlantWildV2 is better than PlantVillage. -> **Action**: In Section 3.1, explicitly mention the "8-pixel background experiment" (lines 177-178 of `main.tex`). Emphasize that while not perfect (12.87%), it is drastically better than the lab-bias benchmark (~49%). This *empirical evidence* is already in the text but needs to be highlighted as the direct answer to this critique.
*   **Feature Extraction Clarity**: "What occurs when two similar classes from different image?" -> **Action**: Clarify in Section 3.3 that features are extracted *per image* and stored independently. The class labels are associated with each feature vector, handling class overlap naturally.
*   **Figure 5 Clarity**: "Explain what the black (100) and yellow (1) lines represent." -> **Action**: Update the caption for `fig:ablation_plot` (or relevant figure) to explicitly state that vertical lines mark specific component thresholds (e.g., $n=100$ as the optimal point).
*   **"Fair Comparison" Defense**: The reviewer claims Experiment 4.4 is unfair because baselines are end-to-end. -> **Action**:
    1.  Refute this by highlighting **Experiment 2** (Table \ref{tab:master_results_summary}), where **ALL** models (ResNet, EfficientNet, YOLO) were used as **feature extractors** under the exact same pipeline. This IS the fair comparison the reviewer asked for, and YOLO won.
    2.  For the end-to-end comparison (Table \ref{tab:benchmark}), frame it not as a "fair competition" but as a "Cost-Benefit Analysis" showing that our cheap method rivals expensive SOTA methods.
*   **Label Engineering Justification**: "Insufficient justification." -> **Action**: Strengthen Section 3.2. Explicitly state that the "Treatment-Based" mapping was validated by a **plant pathology expert** (as mentioned in line 217) and aligns with real-world fungicide application protocols, reducing *clinical* ambiguity even if visual ambiguity remains.

### 5. Tone and Formatting
*   **Output Format**: Provide the **full updated LaTeX code** for the modified sections.
*   **Precision**: Use specific terms (e.g., "Layer 8 C2f block" instead of "an intermediate layer").
*   **Style**: Maintain high-quality academic English.

**Immediate Action:**
Read the `src/` files first to establish the "Ground Truth". Then, edit `main.tex` to align perfectly with that truth.
