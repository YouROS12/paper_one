
\section{Methodology}
\label{sec:methodology}

We designed the methodology to evaluate a two-stage classification pipeline against an end-to-end baseline, with emphasis on the impact of the proposed label engineering strategy. Figure~\ref{fig:system_architecture} shows the workflow from data ingestion to evaluation, and Algorithm~\ref{alg:pipeline} presents the formal steps.

\begin{algorithm}
\caption{Two-Stage Pipeline: Feature Engineering and Classifier Training}
\label{alg:pipeline}
\begin{algorithmic}[1]
\State \textbf{Input:} Raw training images $\mathcal{I}_{train}$, corresponding labels $\mathcal{Y}_{train}$, raw test images $\mathcal{I}_{test}$.
\State \textbf{Output:} A trained and tuned classifier $\mathcal{C}_{final}$, performance metrics (Accuracy, F1-Score).

\Statex
\Statex \textit{//--- Phase 1: Feature Engineering \& Dimensionality Reduction ---}
\State Load pre-trained backbone $\mathcal{M}$ (YOLOv8m, ResNet50, or EfficientNet-B0).
\State Generate feature bank $\mathcal{F}_{train}$ by processing each image in $\mathcal{I}_{train}$ through $\mathcal{M}$ and flattening the activation maps.
\State Generate feature bank $\mathcal{F}_{test}$ from $\mathcal{I}_{test}$ using the same process.
\State Fit compression transformer $\mathcal{T}$ (IPCA or TruncatedSVD) on $\mathcal{F}_{train}$ to obtain latent features.
\State Determine optimal $n$ via ablation (Experiment 1); set $n^{*} = 100$.
\State $\mathcal{X}_{train} \gets \text{Transform}(\mathcal{T}, \mathcal{F}_{train}, n^{*})$
\State $\mathcal{X}_{test} \gets \text{Transform}(\mathcal{T}, \mathcal{F}_{test}, n^{*})$

\Statex
\Statex \textit{//--- Phase 2: Classifier Training \& Evaluation ---}
\State Define a classifier $\mathcal{C}$ (SVC).
\State Train the classifier $\mathcal{C}_{final}$ on the full $\mathcal{X}_{train}$ and $\mathcal{Y}_{train}$.
\State Make predictions on the unseen test data:
\State $\mathcal{Y}_{pred} \gets \mathcal{C}_{final}.\text{predict}(\mathcal{X}_{test})$
\State Evaluate the model by comparing $\mathcal{Y}_{pred}$ with the true test labels $\mathcal{Y}_{test}$.
\State \textbf{return} $\mathcal{C}_{final}$, Performance Metrics
\end{algorithmic}
\end{algorithm}

\begin{landscape}
\begin{figure}
 \centering
  \includegraphics[width=\linewidth,height=0.85\textheight,keepaspectratio]{diagram.png}
  \caption{System architecture of the proposed two-stage classification pipeline.}
  \label{fig:system_architecture}
\end{figure}
\end{landscape}
\subsection{Dataset}

We use the PlantWildV2 dataset, a large-scale benchmark for in-the-wild plant disease recognition. It contains 11,349 images across 115 disease classes, with predefined training and test splits used without modification. Table~\ref{tbl:dataset_description} summarizes key statistics, and Figure~\ref{fig:inter_class} illustrates intra-class variability and inter-class similarity. We selected this dataset to approximate real-world agricultural conditions and avoid limitations of many lab-based datasets.

\begin{figure}
    \centering
    \includegraphics[width=0.9\textwidth]{image.png}
    \caption{Illustration of intra-class variance and inter-class similarity among plant disease images (right), alongside the dataset with the highest volume of in-the-wild images (left).}
    \label{fig:inter_class}
\end{figure}

Performance on in-the-wild data is affected by complex, noisy backgrounds, unlike lab datasets with uniform backgrounds. A critical issue is dataset bias, which can inflate reported performance. This bias is well documented in PlantVillage. In a key study, \cite{noyan2022uncovering} trained a model using only eight background pixels from PlantVillage images and achieved 49.0\% accuracy on the held-out test set, compared to a random-guess baseline of 2.6\%. This shows that spurious background cues correlate with class labels, enabling models to learn shortcuts instead of disease features. We therefore rely on PlantWildV2 to reduce these biases and to develop a model more robust for practical use.

\begin{table}
    \caption{A summary of popular plant disease datasets.}
    \label{tab:popular_datasets}
    \centering
    \begin{tabular}{l c c c l}
        \toprule
        \textbf{Dataset} & \textbf{Plant} & \textbf{Size} & \textbf{Resolution} & \textbf{Setting} \\
        \midrule 
        \cite{fenu2021using} & Pear & 3,505 & Multiple & Field \\	 
        \cite{krohling2019bracol} & Coffee & 4,407 & 2048×1024 & Lab \\	
        \cite{parraga2019rocole} & Coffee & 1,560 & Multiple & Field \\	
        \cite{thapa2020plant} & Apple & 3,651 & 2048×1365 & Field \\			 
        \cite{prajapati2017detection} & Rice & 120 & 2848×4288 & Lab \\	
        \cite{rauf2019citrus} & Citrus & 759 & 256×256 & Lab \\ 
        \cite{hughes2015open} & Multiple & 54,309 & Multiple & Lab \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{table}
\caption{PlantWildV2 Dataset Description.}
\label{tbl:dataset_description}
\centering
\begin{tabular}{lcc}
\toprule
 & Train & Test \\
\midrule
Number of Classes & 115 & 115 \\
Mean images/class & 78.3 & 20.4 \\
Total images & 9,001 & 2,348 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Label Engineering}
Given the high inter-class visual similarity among diseases, we evaluated two label consolidation strategies (Figures~\ref{fig:19_classes} and \ref{fig:11_classes}):  
\begin{enumerate}
    \item \textbf{Visual Grouping:} We mapped the 115 fine-grained classes to 19 super-classes based on shared visual nomenclature (e.g., grouping rust types under `Rust').  
    \item \textbf{Treatment-Based Grouping:} We mapped the 115 classes to 11 actionable super-classes defined by pathogen type and management strategy (e.g., FungalRust, ViralDisease). A plant pathology expert reviewed this consolidation to ensure that merged categories correspond to consistent treatment protocols.
\end{enumerate}
All experiments used the same images; only the target labels differed by grouping strategy.  

\begin{figure}
    \centering
    \includegraphics[width=\textwidth]{19_classes.pdf}
    \caption{19-class Visual-Based Grouping strategy.}
    \label{fig:19_classes}
\end{figure}

\begin{figure}
    \centering
    \includegraphics[width=\textwidth]{11_classes.pdf}
    \caption{11-class Treatment-Based Grouping strategy.}
    \label{fig:11_classes}
\end{figure}

\subsection{Two-Stage Pipeline}
The pipeline has three sequential stages—feature extraction, dimensionality reduction, and machine learning classification—designed to transform raw images into predictions (Figure~\ref{fig:system_architecture}). Decoupling feature extraction from classification supports lightweight retraining without re-optimizing the deep backbone, which is advantageous for scenarios requiring rapid model adaptation and minimal computational cost.

\textbf{Feature extraction.} We compared three state-of-the-art backbones used as frozen feature extractors:
\begin{itemize}
    \item \textbf{ResNet50}: A standard deep residual network widely used in image classification.
    \item \textbf{EfficientNet-B0}: A lightweight architecture optimized for mobile applications.
    \item \textbf{YOLOv8m}: A modern object detection architecture adapted for feature extraction. We extracted activation maps from the final C2f block (Layer 8), selected as a trade-off between semantic depth and spatial resolution.
\end{itemize}
All backbones were pretrained on ImageNet (or COCO in YOLO's case) and fine-tuned on the PlantWildV2 training split. Features were extracted once and stored as a static bank.

\textbf{Reduced Order Modelling via PCA/SVD.} The high-dimensional feature maps extracted from deep backbones contain significant redundancy. We implemented Reduced Order Modelling (ROM) using two techniques: Incremental Principal Component Analysis (IPCA) for memory-efficient streaming, and Truncated Singular Value Decomposition (SVD). We compressed the feature space to a compact latent representation of $n=100$ dimensions, as identified by our ablation study. The transformers were fitted only on the training set to prevent data leakage.

\textbf{Classification.} We trained a Support Vector Classifier (SVC) on the resulting low-dimensional representations. We used scikit-learn's implementation with a radial basis function (RBF) kernel and class weighting to address imbalance.

\subsection{Experimental Design}

We conducted three sequential experiments to identify the optimal pipeline configuration.

\textbf{Experiment 1: Optimal \textit{n\_components} Ablation.}
We aimed to identify the optimal number of principal components. We expected an intermediate dimensionality to retain signal while discarding noise. We trained SVC models with component counts ranging from 100 to 5{,}745 and identified $n=100$ as the optimal trade-off point where overfitting was minimized.

\textbf{Experiment 2: Backbone and Compression Comparison.}
Using the optimal dimensionality (\textbf{n=100}), we conducted a comprehensive comparison of three backbones (YOLOv8m, ResNet50, EfficientNet-B0) combined with two compression methods (IPCA, SVD). This resulted in six model configurations, evaluated on both the 19-class (Visual) and 11-class (Treatment-based) tasks. This experiment directly addresses the need for comparative validation against standard architectures.

\textbf{Experimental 3: Final Model Validation.}
We selected the best-performing configuration (YOLOv8m + IPCA) for a final in-depth evaluation, including per-class analysis and confusion matrix inspection.


\section{Results}
\label{sec:results}
We report results in three parts: (1) an ablation confirming feature dimensionality, (2) a comparative analysis of backbones demonstrating the superiority of the YOLO-based approach, and (3) a detailed evaluation of the champion model.

\subsection{Optimal Feature Dimensionality: Less is More}
Our initial ablation confirmed that performance peaks at \textbf{100 principal components} and degrades with higher dimensionality. This counter-intuitive result indicates that aggressive compression acts as a powerful regularizer, filtering out task-irrelevant background noise often present in high-dimensional feature maps. All subsequent results use $n=100$.

\begin{figure}
  \centering
  \includegraphics[width=0.8\textwidth]{n_components_ablation_plot.png}
  \caption{F1-Macro as a function of the number of principal components. Performance peaks at n=100 and then degrades, indicating that strong dimensionality reduction regularizes the task.}
  \label{fig:ablation_plot}
\end{figure}

\subsection{Backbone and Method Comparison}
Table~\ref{tab:master_results_summary} presents the performance of all tested configurations. The YOLOv8m backbone significantly outperformed ResNet50 and EfficientNet-B0 across all tasks. Specifically, the proposed \textbf{YOLOv8m + IPCA} pipeline achieved a test accuracy of \textbf{86.07\%}, surpassing the EfficientNet-B0 + IPCA pipeline (70.53\%) by over 15 percentage points.

Interestingly, Incremental PCA (IPCA) yielded slightly better or comparable results to Truncated SVD, while offering the practical advantage of stream-processing large datasets without loading the entire feature bank into memory. The treatment-based (11-class) and visual-based (19-class) groupings showed identical high performance with the YOLO backbone, suggesting its features are robust enough to capture the underlying pathology regardless of the specific label hierarchy.

\begin{table}
\centering
\caption{Test performance of different backbones and compression methods (n=100). The YOLOv8m + IPCA pipeline achieves the highest accuracy, significantly outperforming standard architectures.}
\label{tab:master_results_summary}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllccc}
\toprule
\textbf{Labeling Strategy} & \textbf{Backbone} & \textbf{Compression} & \textbf{Accuracy (\%)} & \textbf{F1-Macro} & \textbf{Training Time (s)} \\
\midrule
\multirow{6}{*}{Visual Grouping (19 Classes)} 
 & EfficientNet-B0 & IPCA & 68.23 & 0.705 & 33.5 \\
 & EfficientNet-B0 & SVD & 65.93 & 0.684 & 31.0 \\
 & ResNet50 & IPCA & 66.23 & 0.681 & 34.5 \\
 & ResNet50 & SVD & 59.63 & 0.618 & 32.8 \\
 & \textbf{YOLOv8m (Champion)} & \textbf{IPCA} & \textbf{86.07} & \textbf{0.854} & \textbf{26.3} \\
 & YOLOv8m & SVD & 85.56 & 0.852 & 26.0 \\
\midrule
\multirow{6}{*}{Treatment Grouping (11 Classes)} 
 & EfficientNet-B0 & IPCA & 70.53 & 0.706 & 37.8 \\
 & EfficientNet-B0 & SVD & 67.97 & 0.685 & 36.8 \\
 & ResNet50 & IPCA & 68.27 & 0.679 & 40.4 \\
 & ResNet50 & SVD & 61.07 & 0.619 & 39.9 \\
 & \textbf{YOLOv8m (Champion)} & \textbf{IPCA} & \textbf{86.07} & \textbf{0.854} & \textbf{26.1} \\
 & YOLOv8m & SVD & 85.56 & 0.852 & 25.4 \\
\bottomrule
\end{tabular}%
}
\end{table}

\subsection{In-Depth Analysis of the Champion Pipeline}
We analyzed the champion model (YOLOv8 + IPCA(n=100) + SVC) on the 11-class treatment-based task.

\subsubsection{Per-Class Performance}
The model demonstrates exceptional robustness, particularly on visually distinct classes. Table~\ref{tab:per_class_report} shows F1-scores exceeding 0.90 for \texttt{fungal\_powdery\_mildew} and \texttt{abiotic\_disorder}. Even for challenging classes like \texttt{fungal\_rot\_fruit\_disease}, the model maintains respectable performance.

\begin{table}
\centering
\caption{Per-class performance of the champion SVC model on the 11-class test set.}
\label{tab:per_class_report}
\begin{tabular}{lrrrr}
\toprule
\textbf{Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Support} \\
\midrule
abiotic\_disorder & 0.85 & 1.00 & 0.92 & 23 \\
bacterial\_disease & 0.83 & 0.80 & 0.81 & 320 \\
fungal\_downy\_mildew & 0.81 & 0.90 & 0.85 & 150 \\
fungal\_leaf\_disease & 0.90 & 0.85 & 0.87 & 835 \\
fungal\_powdery\_mildew & 0.95 & 0.94 & 0.95 & 188 \\
fungal\_rot\_fruit\_disease & 0.78 & 0.81 & 0.79 & 152 \\
fungal\_rust & 0.89 & 0.89 & 0.89 & 320 \\
fungal\_scab & 0.86 & 0.98 & 0.92 & 64 \\
fungal\_systemic\_smut\_gall & 0.79 & 0.86 & 0.82 & 96 \\
oomycete\_lesion & 0.80 & 0.90 & 0.85 & 15 \\
viral\_disease & 0.81 & 0.88 & 0.84 & 185 \\
\midrule
\textbf{Macro Avg} & \textbf{0.84} & \textbf{0.89} & \textbf{0.86} & \textbf{2,348} \\
\textbf{Weighted Avg} & \textbf{0.86} & \textbf{0.86} & \textbf{0.86} & \textbf{2,348} \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Error Analysis and Model Calibration}
The confusion matrix (Figure~\ref{fig:final_cm}) confirms low misclassification rates. The reliability diagram (Figure~\ref{fig:reliability_diagram}) indicates the model remains well-calibrated, a critical property for automated decision support systems in agriculture.

\begin{figure}
  \centering
  \includegraphics[width=0.75\textwidth]{confusion_matrix_yolo_ipca.png} 
  \caption{Normalized confusion matrix for the champion YOLO+IPCA model.}
  \label{fig:final_cm}
\end{figure}

\begin{figure}
\centering
\includegraphics[width=0.65\textwidth]{reliability_diagram_sklearn_SVC.png}
\caption{Reliability diagram for the champion model.}
\label{fig:reliability_diagram}
\end{figure}

\subsection{Comparative Analysis: Classification Performance vs. Training Complexity}

To validate the proposed method, we compared it against a fully fine-tuned EfficientNet-B0 baseline. The baseline model was trained end-to-end for 10 epochs on an NVIDIA A100 GPU, requiring approximately 5.8 hours.

As shown in Table~\ref{tab:benchmark}, our proposed pipeline achieves a test accuracy of \textbf{86.07\%}, which not only matches but **surpasses** the end-to-end baseline (82.50\%). This result challenges the prevailing assumption that end-to-end fine-tuning is strictly necessary for high performance. By leveraging the strong feature extraction capabilities of YOLOv8 and the regularization effect of PCA(n=100), we achieve superior generalization.

Moreover, the computational advantage is decisive. While the baseline requires heavy GPU resources for backpropagation, our pipeline's classifier can be retrained in just \textbf{26.1 seconds} on a standard CPU. This represents a \textbf{~800$\times$ speedup} in adaptation time, making the system uniquely suitable for dynamic agricultural environments where new diseases may require frequent model updates.

\begin{table}[h]
\centering
\caption{Comparative Analysis: Classification Performance vs. Computational Cost. The proposed ROM framework outperforms the end-to-end baseline while reducing training time by orders of magnitude.}
\label{tab:benchmark}
\begin{tabular}{lcccc}
\hline
\textbf{Model} & \textbf{Test Acc.} & \textbf{Training Time} & \textbf{Hardware} & \textbf{Retraining Cost} \\ \hline
EfficientNet-B0 (Fine-Tuned) & 82.50\% & $\sim$5.8 hours & GPU (A100) & High (Backprop) \\
\textbf{Proposed Pipeline (YOLO+IPCA)} & \textbf{86.07\%} & \textbf{26.1 seconds} & \textbf{CPU} & \textbf{Negligible} \\ \hline
\end{tabular}
\end{table}
