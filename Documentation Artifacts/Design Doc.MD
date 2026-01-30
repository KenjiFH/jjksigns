# RFC: Geometric Hand Gesture Recognition Architecture

**Status:** Proposed
**Date:** 2026-01-29
**Author:** [Kenji F.]

## Context
The objective is to classify static and near-static two-hand gestures from live camera input. Existing solutions often rely on raw image processing, which introduces high latency and environmental sensitivity. We require a system that prioritizes stability, predictability, and interpretability over raw computer vision research. The system must operate in real-time on consumer hardware without depth sensors, and distinctly categorize inputs into Positive (Gesture), Negative (Explicit Reject), or Unidentified (Low Confidence).

## Decision
We will implement a **modular, geometry-driven classification pipeline** that separates perception (MediaPipe) from intent classification (Scikit-Learn).

### 1. System Architecture
The pipeline is strictly defined as follows:
1.  **Perception Layer:** MediaPipe Hands extracts 21 3D landmarks per hand from the raw RGB stream.
2.  **Feature Abstraction:** Raw landmarks are converted into a normalized, view-invariant feature vector (129 features) incorporating joint angles, relative positions, and wrist-anchored coordinates.
3.  **Classification:** A Support Vector Machine (SVM) with an RBF kernel classifies the geometric vector.
4.  **Runtime Logic:** A post-processing layer applies confidence thresholds to determine gesture intent.

### 2. Component Selection
* **Perception:** **MediaPipe**. Selected for its state-of-the-art CPU performance and semantic output. It allows the system to operate in a clean geometric space rather than pixel space.
* **Machine Learning:** **Scikit-Learn (SVM)**. Selected over deep learning frameworks. The problem space (structured, low-dimensional data) favors classical ML for its determinism and ease of deployment.
* **Input Hardware:** **Standard Webcam**. No depth sensors or IR cameras are required.

### 3. Development Workflow
* **Offline Training (Jupyter):** Model development, data visualization, and hyperparameter tuning occur exclusively in Jupyter Notebooks to facilitate exploratory data analysis (EDA).
* **Runtime Inference (Python Module):** Production inference uses pure Python modules loading serialized model artifacts. Notebooks are explicitly barred from the production runtime.

## Alternatives Considered

### Custom Deep Learning Vision Models (CNNs)
* **Status:** Rejected.
* **Reasoning:** Training a custom CNN from scratch requires massive datasets to achieve invariance to lighting, background, and skin tone. MediaPipe abstracts these variations instantly, allowing us to focus on geometry rather than perception.

### Deep Learning Classifiers (PyTorch/TensorFlow)
* **Status:** Rejected.
* **Reasoning:** For a feature vector of ~129 floats, a neural network is unnecessary complexity. It introduces overfitting risks and reduces interpretability compared to an SVM. Deep learning will only be reconsidered if temporal sequence modeling (RNNs/LSTMs) becomes a requirement.

### Python Scripts for Training
* **Status:** Rejected.
* **Reasoning:** While better for version control, scripts inhibit the rapid visual feedback loops required for analyzing confusion matrices and decision boundaries. Jupyter is retained for the *research* phase, provided artifacts are properly serialized.

## Rationale
1.  **Separation of Concerns:** By offloading the "vision" problem to MediaPipe, we reduce the domain problem to simple geometric classification. This modularity makes debugging trivial—if landmarks are wrong, it's a MediaPipe issue; if landmarks are right but classification is wrong, it's an SVM issue.
2.  **Invariance by Design:** The feature extraction layer mathematically enforces invariance to scale (bone length normalization) and position (wrist anchoring). We solve these problems with algebra, not data augmentation.
3.  **Resource Efficiency:** The combination of MediaPipe and SVM runs comfortably on CPU, ensuring low latency and high portability.

## Consequences

### Positive
* **Interpretability:** Feature vectors are human-readable (e.g., "distance between thumb and index").
* **Stability:** The system is immune to background noise that does not mimic a hand.
* **Portability:** Zero dependency on GPU hardware for inference allows deployment on diverse consumer devices.

### Negative
* **Perception Bottleneck:** The system is hard-capped by MediaPipe's limitations. If lighting is poor, MediaPipe may fail to detect hands or hallucinate hands from face landmarks. The downstream classifier cannot recover from these upstream errors.
* **Static Limitation:** The current architecture is optimized for static poses. Dynamic gestures requiring velocity or trajectory analysis would require a significant re-architecture of the feature extraction layer.
