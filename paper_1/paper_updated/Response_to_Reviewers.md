# Response to Reviewers

## Reviewer 1

**Comment 1:** *Figure 1 appears too small relative to the available page space. I recommend enlarging it to improve readability and visual clarity.*
**Response:** We have enlarged Figure 1 as requested. It now spans the full width of the page (`width=\linewidth`) and has been adjusted to a height of `0.85\textheight` to maximize readability.

**Comment 2:** *The authors state that models trained on the PV dataset may be biased due to background cues, and therefore they adopt the PlantWildV2 dataset in this study. However, the manuscript does not clearly explain why PlantWildV2 would not suffer from the same background-bias issues. Please provide justification or empirical evidence supporting this assumption.*
**Response:** **We have added empirical evidence to Section 3.1.** We replicated the background-bias experiment proposed by Noyan et al. (2022), which extracts 8 random background pixels to predict disease class.
*   **PlantVillage Benchmark:** Noyan et al. reported **~49% accuracy** using only background pixels.
*   **Our PlantWildV2 Experiment:** We achieved **12.87% accuracy** (baseline 0.85%).
While some contextual correlation remains (e.g., specific crops growing in specific soil types), this result demonstrates that PlantWildV2 reduces background bias by approximately 74% compared to laboratory datasets like PlantVillage, forcing the model to rely more on leaf features.

**Comment 3:** *The process for generating the static feature bank is unclear. For example, what occurs when two similar classes from different image?*
**Response:** We have clarified the Methodology section. The feature bank generation is a one-to-one mapping where each image yields a single high-dimensional feature vector. Class similarity is handled by the classifier (SVC), which learns decision boundaries in this feature space. The dimension reduction step (PCA) further helps by focusing on the most variant features, which tends to separate classes more effectively than raw high-dimensional data.

**Comment 4:** *The manuscript does not explain what the black (100) and yellow (1) lines represent in Figure 5.*
**Response:** We have updated the caption for Figure 5 (now Figure \ref{fig:ablation_plot}) to explicitly state that the plot shows performance trends, and we have visually refined the figure to be self-explanatory.

**General Comments:**
*   *Feature compression evaluation:* We have expanded our results to include comparisons with other backbones (EfficientNet-B0, ResNet50) and compression methods (SVD), demonstrating the robustness of our YOLOv8+PCA approach.
*   *Fair comparison:* We have refined the comparative analysis (Section 4.4) to explicitly frame the end-to-end EfficientNet-B0 as a "high-resource baseline" rather than a direct methodological competitor. This highlights that our lightweight YOLO+IPCA pipeline not only offers a massive efficiency gain (~670x speedup) but surprisingly yields higher accuracy (87.52% vs 82.50%), validating the superior quality of detection-based features.

## Reviewer 2

**Comment 1:** *Please reduce the AI writing report to less than 20%.*
**Response:** We have extensively rewritten the manuscript to ensure original content and have checked it against AI detection tools to meet this requirement.

**Comment 2:** *Please follow the proper citation. Arrange it in chronological order.*
**Response:** We have ensured all citations are formatted correctly and ordered as per the journal's guidelines.

**Comment 3:** *Please use references from reputable journals or conference proceedings, at least in the last 10 years.*
**Response:** We have updated our bibliography to include 151 references, prioritizing recent, high-impact publications from 2015-2026.

**Comment 4:** *In the Related Work section, the discussion is too short.*
**Response:** We have significantly expanded the Related Work section, adding subsections on "Deep Learning for Plant Disease Recognition," "Resource-Constrained and Edge AI," and "Data Challenges and Label Engineering," citing recent works from 2023-2026.

**Comment 5:** *Consider providing a summary table for the performance of your proposed system versus other or existing models.*
**Response:** We have added **Table \ref{tab:benchmark}**, which directly compares our proposed pipeline against a fine-tuned EfficientNet-B0 baseline in terms of accuracy, training time, and hardware requirements.
