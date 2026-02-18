
You are acting as the **Lead Academic Editor** for a research paper on **AI in Plant Pathology**. Your task is to synchronize the LaTeX manuscript (`paper_updated/paper_exported_from_overleaf/main.tex`) with the final version of the experimental codebase (`src/`).

**Input Files to Read:**
1.  `src/preprocessing.py`: Defines the data processing pipeline, feature extraction (YOLO layer 8), and label engineering (11-class mapping).
2.  `src/training_and_evaluation.py`: Defines the classifier training (SVC settings), evaluation metrics, and experimental loop.
3.  `src/plot_results.py`: Generates the figures (confusion matrices, bar charts). Use this to verify figure captions and descriptions.
4.  `archive/final_exp_plantwild_cusvm.py`: **Context for Model Choice**. This shows our preliminary exploration. We settled on SVC to demonstrate that **YOLOv8 features are so robust** that even a standard classifier yields high performance. The focus is on **Backbone Quality**, not Classifier Superiority.
5.  `paper_updated/paper_exported_from_overleaf/biblio.bib`: The bibliography file. You will need to **add references** here.
6.  `paper_updated/paper_exported_from_overleaf/main.tex`: The LaTeX source file that needs to be updated.

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

### 4. Tone and Formatting
*   **Output Format**: Provide the **full updated LaTeX code** for the modified sections, or the entire file if requested.
*   **Precision**: Use specific terms (e.g., "Layer 8 C2f block" instead of "an intermediate layer").
*   **Style**: Maintain high-quality academic English.

### 5. Reviewer 2 Addressal (Major Revisions)
*   **AI Writing Reduction**: Rewrite the **Introduction** and **Abstract** to sound more natural and less "AI-generated". Use varied sentence structures.
*   **Citation Style**: specific instructions:
    *   Use `\citet{}` or `\citep{}` for `elsarticle-harv` style.
    *   Ensure citations are **chronological** within parentheses (e.g., `(Smith 2015; Jones 2018)`).
*   **New Content to Add**:
    1.  **Related Work Expansion**: Add subsections for "Deep Learning in Plant Pathology" (CNNs/Transformers 2015-2024) and "Dimensionality Reduction" (PCA/SVC theory).
    2.  **Label Engineering**: Add a subsection in Methodology explaining the *agronomical* reason for 11 classes (fungicide treatments).
    3.  **Dataset (Section 3.1)**: Add the background bias numbers (12.87% vs ~49%).
    4.  **Algorithm 1**: Add text specifically explaining the steps in Algorithm 1, including Big-O complexity ($O(d \cdot n)$).
    6.  **Metrics**: Define Precision, Recall, F1, Accuracy formally.
    7.  **Statistical Rigor**: Mention that results are derived from robust protocols (e.g., standard splits). If the code supports it (check `final_exp_plantwild_cusvm.py`), mention the use of multiple random seeds to ensure validity.
    8.  **Methods Documentation**: Ensure Algorithm 1 is described *step-by-step* in the text, maximizing reproducibility.
*   **Bibliography**:
    *   **Action**: You MUST expand `biblio.bib` to include at least **50 references**.
    *   **Source**: Use your internal knowledge to generate correct BibTeX entries for reputable Plant Pathology/DL papers from journals like *Computers and Electronics in Agriculture*, *IEEE Access*, *Frontiers in Plant Science* (2014-2024).

**Immediate Action:**
Read the `src/` files first to establish the "Ground Truth". Then, edit `main.tex` to align perfectly with that truth.
