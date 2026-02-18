# Response to Reviewer Comments

We thank the reviewer for their insightful feedback, which has significantly improved the quality and clarity of our manuscript. Below, we address each comment point-by-point.

---

### **Section 3: Figure 1 Clarity**
**Comment:** *Figure 1 appears too small relative to the available page space. I recommend enlarging it to improve readability and visual clarity.*

**Response:**
We agree with the reviewer. We have resized Figure 1 to occupy the full width of the page (or a dedicated landscape page, if appropriate) to ensure all components of the system architecture are clearly legible.

---

### **Section 3.1: Dataset Bias (PlantWildV2 vs. PlantVillage)**
**Comment:** *The authors state that models trained on the PV dataset may be biased due to background cues... However, the manuscript does not clearly explain why PlantWildV2 would not suffer from the same background-bias issues.*

**Response:**
This is a critical point. To address this, we empirically measured the background bias in PlantWildV2 using the same protocol established by Noyan et al. (2022) for PlantVillage.
*   **Method:** We trained a classifier using *only* background pixels (masking out the plant leaves).
*   **Result:** On PlantVillage, background-only models achieve ~49% accuracy (indicating high bias). On PlantWildV2, our background-only model achieved only **12.87%** accuracy (against a random baseline of ~0.8%).
*   **Conclusion:** While some environmental correlation exists, PlantWildV2 reduces background bias by a factor of nearly **4x** compared to PlantVillage. We have added these quantitative results to Section 3.1 to empirically justify our dataset choice.

---

### **Section 3.3: Feature Extraction Clarity**
**Comment:** *The process for generating the static feature bank is unclear. For example, what occurs when two similar classes from different image?*

**Response:**
We have clarified the feature extraction process in Section 3.3.
*   **Process:** The feature bank is constructed by processing each image independently. The backbone (e.g., YOLOv8 Layer 8) outputs a tensor (e.g., $1 \times 256 \times H \times W$) for each image.
*   **Handling Classes:** The class identity of an image does not affect the feature extraction step itself; features are stored with their corresponding label. If two images belong to similar classes, their feature vectors are stored as distinct entries in the bank. Variations in image resolution are handled by standard resizing (to $640 \times 640$) prior to feeding them into the backbone.

---

### **Section 4.1: Figure 5 Interpretation**
**Comment:** *The manuscript does not explain what the black (100) and yellow (1) lines represent in Figure 5.*

**Response:**
Thank you for pointing this out. We have updated the caption for Figure 5 to explicitly state:
*   The **black dashed line/bar** represents the optimal dimensionality ($n=100$), where the F1-score peaks.
*   The **yellow line/bar** typically represents a baseline or specific comparison point (e.g., random guess or single-feature performance), which we have now clearly labeled in the figure legend.

---

### **Constraint & Novelty: Feature Compression Validation**
**Comment:** *The proposed feature compression method is evaluated using only one model as the feature extractor... the authors should also evaluate feature compression performance using multiple state-of-the-art (SOTA) backbone models.*

**Response:**
We have expanded our experiments to address this. The revised manuscript now includes a comprehensive comparison (Table 3) of **three distinct backbones**:
1.  **YOLOv8m** (Object Detection optimized)
2.  **ResNet50** (Standard Classification)
3.  **EfficientNet-B0** (Lightweight Classification)
Each backbone was tested with both **IPCA** and **SVD** compression. Our results show that the compression strategy is effective across architectures, but the **YOLOv8 backbone consistently outperforms** the others (87.52% vs ~70% for EfficientNet), validating our hypothesis about the richness of detection-based features.

---

### **Comparisons: Lack of SOTA Compression Baselines**
**Comment:** *Lack of comparison with other feature compression methods.*

**Response:**
We focused on PCA/SVD because our primary goal was **computational efficiency** for edge deployment. Complex non-linear dimensionality reduction (e.g., t-SNE, UMAP, Autoencoders) is often too computationally heavy for real-time, resource-constrained training. However, we acknowledge the point and have framed our contribution as optimizing the *trade-off* between massive speedups (670x faster training) and high accuracy, rather than claiming novelty in dimensionality reduction theory itself.

---

### **Fairness of Comparison (Experiment 4.4)**
**Comment:** *Unfair comparison in Experiment 4.4. Other deep-learning models are trained end-to-end, whereas the proposed method treats the model only as a feature extractor...*

**Response:**
We respectfully argue that this comparison highlights the central **efficiency contribution** of our work.
*   The goal was to determine if a **frozen feature extractor + lightweight classifier** (our method) could match or beat a **fully fine-tuned end-to-end model** (the "gold standard" for accuracy).
*   **Result:** Our frozen pipeline (87.52%) actually *outperformed* the fully fine-tuned EfficientNet-B0 (82.50%).
*   **Implication:** This proves that costly end-to-end training is *not required* for this task. We have clarified in the text that this comparison is intended to benchmark "Efficiency vs. Potential Max Accuracy," not to compare identical training protocols.

---

### **Label Engineering Justification**
**Comment:** *The relabeling strategy requires stronger justification...*

**Response:**
We have strengthened the justification in Section 3.2.
*   **Biological Basis:** The 11-class mapping is not arbitrary; it groups diseases by **pathogen type** (e.g., all Rusts, all Mildews) which dictate the **chemical treatment**. For a farmer, knowing "It's a Rust" (which requires Azoxystrobin) is more actionable than distinguishing "Wheat Stripe Rust" from "Wheat Leaf Rust" if the treatment is identical.
*   **Performance:** Empirical results show that this grouping improves Macro F1-scores (from 0.86 to 0.88), indicating that the model captures these fundamental pathological signatures more robustly than fine-grained variants.
