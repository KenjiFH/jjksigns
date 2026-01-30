# Design Doc: Geometric Hand Gesture Recognition System

**Author:** [Kenji F.]
**Status:** Draft
**Scope:** Architecture and Implementation Strategy

## 1. System Overview
This system provides real-time classification of static and quasi-static two-hand gestures. It abstracts raw camera input into geometric landmark data to classify user intent.

**Core Principles:**
* **Geometry over Pixels:** Classification relies on normalized vector math (angles, relative distances) rather than texture or color.
* **Explicit Rejection:** The system distinguishes between *Positive* (recognized), *Negative* (explicit reject), and *Unidentified* (low confidence).
* **Stability:** Priorities are predictability and low latency over per-frame maximization.

## 2. Architecture
The pipeline separates perception (MediaPipe), feature engineering, and classification (SVM).

```text
[Webcam] 
   ↓ Raw RGB
[MediaPipe Hands] 
   ↓ 3D Landmarks (21 per hand)
[Feature Extractor] 
   ↓ Normalized Vector (129 features)
[SVM Classifier] 
   ↓ Class Probabilities
[Decision Logic]
   ↓ 
[Final Gesture Intent]
