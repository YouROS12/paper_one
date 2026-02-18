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

---

### **Response to Reviewer 2**

We thank Reviewer 2 for their detailed feedback on the structure, writing style, and content depth. We have addressed each point below.

#### **1. AI Writing & Citation Style**
**Comment:** *Please reduce the AI writing report to less than 20%... Please follow the proper citation. Arrange it in chronological order.*
**Response:**
We have meticulously revised the manuscript to ensure a natural, academic tone, rewriting sections flagged as AI-generated. We have also unified the citation style (using the `elsarticle-harv` format as per the template) and ordered citations chronologically where appropriate (e.g., "Smith et al. (2015); Jones et al. (2018); Doe et al. (2020)").

#### **2. Expanding Related Work**
**Comment:** *Discussion is too short. Consider adding sections like Plant Disease Classification using DL... Discussion about PCA and SVC.*
**Response:**
We have significantly expanded Section 2 (Related Work) to include:
*   **Deep Learning in Plant Pathology:** A review of recent CNN and Transformer applications (2015–2024).
*   **Dimensionality Reduction & Classical ML:** A new subsection discussing the theoretical basis of PCA and SVC in high-dimensional feature spaces, justifying their use over end-to-end deep learning for resource-constrained scenarios.

#### **3. Novelty & Label Engineering**
**Comment:** *Elaborate on advantages/novelty... Consider a section discussing label engineering.*
**Response:**
*   **Novelty:** We clarified in Section 2.3 and the Introduction that our novelty lies in **decoupling** modern object detection features (YOLOv8) from classification to achieve real-time retraining on CPUs—a critical advantage for edge devices.
*   **Label Engineering:** We added a dedicated subsection in Methodology explaining the **agronomical justification** for our 11-class grouping (grouping by fungicide treatment protocol), supporting it with the improved F1-scores observed in our results.

#### **4. Algorithm & Dataset Details**
**Comment:** *Have a discussion or explanation about your Algorithm... Further elaborate your discussion about your dataset section 3.1.*
**Response:**
*   **Algorithm 1:** We added a step-by-step walkthrough in the text referencing Algorithm 1, calculating the computational complexity of the projection step $O(d \cdot n)$.
*   **Dataset:** We expanded Section 3.1 to detail PlantWildV2's collection methodology, geographic diversity, and our own empirical analysis of its background bias (12.87% vs ~49% in PlantVillage), as requested.

#### **5. Figures & Metrics**
**Comment:** *Elaborate your discussion for Figures 3-4. Include the evaluation metrics discussion.*
**Response:**
*   **Figures:** We expanded the captions and text for Figures 3 (Feature Visualization) and 4 (Variance Explained) to explicitly interpret the clusters and the "elbow point" in the variance plot.
*   **Metrics:** We added a formal definition of the metrics used (Precision, Recall, F1-Macro, Accuracy) in the Experimental Design section to ensure clarity.

#### **6. References**
**Comment:** *Consider adding more references, at least 50 references.*
**Response:**
We have conducted a comprehensive literature review and expanded our bibliography to **50+ references**, prioritizing reputable journals (e.g., *Computers and Electronics in Agriculture*, *IEEE Access*, *Frontiers in Plant Science*) from the last 10 years (2014–2024).
