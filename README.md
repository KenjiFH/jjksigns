# Domain Expansion: Real-Time Gesture Recognition with MediaPipe & SVMs

### A Computer Vision & HCI Project inspired by Jujutsu Kaisen


## 📖 Overview
This project is a real-time gesture recognition engine that triggers visual effects (like *Domain Expansions* from JJK) based on specific hand signs. 

Instead of using heavy Deep Learning models, this project demonstrates an efficient **Machine Learning Pipeline** using **Google MediaPipe** for landmark extraction and a **Support Vector Machine (SVM)** for classification. This approach ensures low latency and high accuracy even on standard CPU hardware.

## ✨ Features
* **Real-time Hand Tracking:** Uses MediaPipe to track 21 3D landmarks per hand.
* **Custom ML Pipeline:** Features a built-in data collection mode to create your own datasets.
* **Scale Invariance:** Custom vector normalization logic ensures gestures work at any distance from the camera.
* **Visual Effects engine:** Triggers complex overlays (Infinite Void, Malevolent Shrine) upon high-confidence detection.
* **HCI Principles:** Implements hysteresis and confidence thresholds to prevent UI flickering.

## 🛠️ Tech Stack
* **Python 3.11**
* **OpenCV:** Video capture and visual overlay rendering.
* **MediaPipe:** Hand landmark detection.
* **Scikit-Learn:** SVM (RBF Kernel) for gesture classification.
* **NumPy:** Vector math and normalization.

## 🚀 Installation

1.  **Clone the repo**
    ```bash
    git clone [https://github.com/KenjiFH/jjksigns.git](https://github.com/KenjiFH/jjksigns.git)
    cd jjksigns
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    python src/camera_input.py
    ```

## 🎮 Controls & Usage

| Key | Action |
| :--- | :--- |
| **ESC** | Quit the application |
| **TAB** | Cycle through available gesture labels (for recording,  |
| **Hold 'C'** | Record training data for the selected label, 2 handed gestures are automatically recorded so no need for a third hand |

### Creating Your Own Gestures (The Pipeline)
This project includes a full training loop. To add a new gesture:

1.  Run the app.
2.  Press `TAB` to select a label (or add a new one in `CLASS_NAMES` in the code).
3.  Hold **'C'** while performing the gesture to capture ~50-100 frames of data.
4.  The data is saved to `models/keypoint_classifier/keypoints.csv`.
5.  Open `notebooks/train_model.ipynb` and run the cells to retrain the SVM.
6.  The new model will be saved as `gesture_model.pkl`. Restart the app to use it!

## 📂 Project Structure
```text
jjksigns/
├── models/
│   ├── hand_landmarker.task   # MediaPipe base model
│   └── keypoint_classifier/   # CSV datasets
├── notebooks/
│   ├── gesture_model.pkl      # Trained SVM model
│   └── train_model.ipynb      # Training workflow
├── src/
│   └── camera_input.py        # Main application entry point
└── requirements.txt
