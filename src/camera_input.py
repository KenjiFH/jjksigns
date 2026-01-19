import mediapipe as mp
import cv2
import math
import numpy as np
import pickle
import csv
import os

#for stat pattern
import random
from abc import ABC, abstractmethod

# ==============================
# CONFIGURATION & CONSTANTS
# ==============================
from pathlib import Path

# 1. Get the directory where THIS script is located
# .resolve() converts it to an absolute path so it's robust
# .parent gives you the folder containing the file
SCRIPT_DIR = Path(__file__).resolve().parent.parent

# 2. Construct paths relative to the script location
# logic: script_dir / "folder" / "filename"

MODEL_PATH = SCRIPT_DIR / "models" / "hand_landmarker.task"
PICKLE_MODEL_PATH = SCRIPT_DIR / "notebooks" / "gesture_model.pkl"
CSV_FILE = SCRIPT_DIR / "models" / "keypoint_classifier" / "keypoints.csv"

# Optional: Print to verify on run
print(f"Loading model from: {MODEL_PATH}")

CLASS_NAMES = [
    ("ThumbsUp", 1),      # Must see exactly 1 hand
    ("Peace", 1),         # Must see exactly 1 hand
    ("Gojo_Void", 1),     # Must see exactly 2 hands (User note: code says 1 here, but label implies specific gesture)
    ("Sukuna_Shrine", 2), # Must see exactly 2 hands
    ("Cinema", 2)
]

# MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# ==============================
# HELPER FUNCTIONS: Math & Features
# ==============================
def hand_to_feature_vector(hand):
    """
    hand: list of 21 NormalizedLandmark objects
    returns: list[float] length 63
    """
    if hand is None or len(hand) != 21:
        raise ValueError("Expected 21 hand landmarks")

    # 1. Translation normalization (wrist anchor)
    wrist = hand[0]
    coords = []
    for lm in hand:
        coords.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

    # 2. Scale invarience (Reference Length Normalization)
    ref = coords[9]  # middle finger landmarks
    scale = math.sqrt(ref[0]**2 + ref[1]**2 + ref[2]**2) + 1e-6 

    coords = [[c / scale for c in p] for p in coords]

    # 3. Flatten
    feature_vector = []
    for p in coords:
        feature_vector.extend(p)

    return feature_vector

def merge_two_hands(result):
    """
    result: MediaPipe HandLandmarkerResult
    returns: merged feature vector (Left | Right | cross-hand)
    """
    hands = {}
    for idx, hand in enumerate(result.hand_landmarks):
        handedness_info = result.handedness[idx][0]
        handedness = handedness_info.category_name  
        hands[handedness] = hand

    # ---- Per-hand features ----
    if "Left" in hands:
        left_feat = hand_to_feature_vector(hands["Left"])
        left_wrist = np.array([hands["Left"][0].x, hands["Left"][0].y, hands["Left"][0].z])
    else:
        left_feat = [0.0] * 63
        left_wrist = None

    if "Right" in hands:
        right_feat = hand_to_feature_vector(hands["Right"])
        right_wrist = np.array([hands["Right"][0].x, hands["Right"][0].y, hands["Right"][0].z])
    else:
        right_feat = [0.0] * 63
        right_wrist = None

    # ---- Cross-hand features ----
    if left_wrist is not None and right_wrist is not None:
        wrist_dist = np.linalg.norm(left_wrist - right_wrist)
    else:
        wrist_dist = 0.0

    # Optional presence flags
    presence = [int(left_wrist is not None), int(right_wrist is not None)]

    return left_feat + right_feat + [wrist_dist] + presence

# ==============================
# HELPER FUNCTIONS: Initialization
# ==============================
def load_gesture_model():
    try:
        with open(PICKLE_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
        return model
    except FileNotFoundError:
        print("Warning: Model not found. Running in Data Collection mode only.")
        return None

def initialize_csv():
    if not os.path.exists(CSV_FILE):
        # 129 features = 63 (Left) + 63 (Right) + 1 (Dist) + 2 (Presence)
        header = [f"f_{i}" for i in range(129)] + ["label"]
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

# ==============================
# HELPER FUNCTIONS: Visualization
# ==============================
def draw_hand(frame, hand_landmarks):
    h, w, _ = frame.shape
    # Draw landmarks
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # Draw skeleton
    for start, end in HAND_CONNECTIONS:
        x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
        x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


# Strategy pattern Blueprint, for defining new gestures
class VisualEffect(ABC):
    def __init__(self, duration=30):
        self.max_duration = duration
        self.timer = 0

    def start(self):
        """Resets the timer to start the effect."""
        self.timer = self.max_duration

    @property
    def is_active(self):
        return self.timer > 0

    def update(self):
        """Tick down the timer."""
        if self.timer > 0:
            self.timer -= 1

    @abstractmethod
    def render(self, frame):
        """Draw the effect on the frame. Must be implemented by children."""
        pass
#TODO get rid of this spagetti if else code with strategy pattern (ABC, abstractmethod)
def apply_visual_effects(frame, gesture_name, confidence, active_effect, effect_timer):
    """
    Handles the state logic for Gojo/Sukuna effects.
    Returns: (frame, updated_active_effect, updated_effect_timer)
    """
    # Trigger Logic
    if confidence > 0.90:
        if gesture_name[0] == "Gojo_Void":
            active_effect = "GOJO"
            effect_timer = 30  
        elif gesture_name[0] == "Sukuna_Shrine":
            active_effect = "SUKUNA"
            effect_timer = 30
        else:
            active_effect = None
            effect_timer = 0

    if gesture_name[0] == "None" or confidence < 0.50: 
        active_effect = None
        effect_timer = 0


    # Render Logic
    if effect_timer > 0:
        overlay = np.zeros_like(frame)
        h, w, _ = frame.shape
        
        if active_effect == "GOJO":
             # --- GOJO EFFECT---
            # --- 1. "Information Overload" (Inverted Colors) ---
            # We blend the normal frame with an inverted version of itself.
            # This creates a "negative" ghosting effect that mimics the anime's visual style.
            inverted_frame = cv2.bitwise_not(frame)
            frame = cv2.addWeighted(frame, 0.6, inverted_frame, 0.4, 0) 

            # --- 2. The "Void" Atmosphere (Deep Purple + Stars) ---
            overlay = np.zeros_like(frame)
            overlay[:] = (255, 50, 100) # Deeper, darker Violet (BGR) instead of hot pink
            
            # --- 3. Dynamic Stars (The "Infinite" part) ---
            # Draw 50 random white dots every frame to simulate moving through space
            
            for _ in range(50):
                star_x = random.randint(0, w)
                star_y = random.randint(0, h)
                star_size = random.randint(1, 3) # Tiny white dots
                cv2.circle(overlay, (star_x, star_y), star_size, (255, 255, 255), -1)
                
            # Blend the Starry Purple Overlay onto the Inverted Frame
            frame = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)

            # Text
            text = "DOMAIN EXPANSION: UNLIMITED VOID"
            color = (255, 255, 255)
        elif active_effect == "SUKUNA":
            # --- SUKUNA EFFECT: Red Tint + Rapid Slashes ---
            
            # Background Red Tint
            overlay[:] = (0, 0, 200)
            frame = cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)
            
            # Draw Rapid Slashes
            #  draw 5-10 random lines per frame to simulate rapid cuts
            num_slashes = random.randint(5, 10) 
            
            for _ in range(num_slashes):
                # Randomize start and end points
                # Case A: Horizontal/Diagonal Slash
                if random.choice([True, False]): 
                    x1, x2 = 0, w
                    y1 = random.randint(0, h)
                    y2 = random.randint(0, h)
                # Case B: Vertical/Diagonal Slash
                else:
                    x1 = random.randint(0, w)
                    x2 = random.randint(0, w)
                    y1, y2 = 0, h

                # Draw the line (White, variable thickness)
                thickness = random.randint(1, 4)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness)

            text = "DOMAIN EXPANSION: MALEVOLENT SHRINE"
            color = (0, 0, 255) # Red text
        else:
            text = ""
            color = (255,255,255)

        if text:
            font = cv2.FONT_HERSHEY_DUPLEX
            scale, thickness = 1.0, 2
            (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
            cv2.putText(frame, text, (w - text_w - 20, 100), font, scale, color, thickness)

        print(f" gesture {gesture_name[0]} count: {effect_timer} for effect {active_effect}")
        effect_timer -= 1
    else:
        active_effect = None

    return frame, active_effect, effect_timer

def draw_ui_overlay(frame, status_text, color, confidence, target_label, class_index):
    h, w, _ = frame.shape 
    base_y = h - 400

    # Prediction Text
    cv2.putText(frame, status_text, (20, base_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    # Confidence Bar
    bar_y_start = base_y + 10
    bar_y_end = base_y + 30
    bar_width = int(confidence * 200)
    
    cv2.rectangle(frame, (20, bar_y_start), (20 + bar_width, bar_y_end), color, -1)
    cv2.rectangle(frame, (20, bar_y_start), (220, bar_y_end), (255, 255, 255), 2)

    # Instructions
    cv2.putText(frame, f"Training Current Label: {target_label} at index: {class_index} (Press TAB to switch)", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Hold 'c' to record Frame", (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# ==============================
# HELPER FUNCTIONS: Logic
# ==============================
def perform_inference(model, merged_features):
    """
    Returns: (gesture_name_tuple, confidence, status_text, color)
    """
    if model is None:
        return ("None", 0), 0.0, "Model Not Loaded", (100, 100, 100)

    input_data = np.array(merged_features).reshape(1, -1)
    probs = model.predict_proba(input_data)[0]
    max_idx = np.argmax(probs)
    confidence = probs[max_idx]
    gesture_name = CLASS_NAMES[max_idx]

    CONFIDENCE_THRESHOLD = 0.90  
    if confidence > CONFIDENCE_THRESHOLD:
        status_text = f"{gesture_name}: {confidence:.2f}"
        color = (0, 255, 0) 
    else:
        status_text = f"Uncertain ({gesture_name}? {confidence:.2f})"
        color = (0, 165, 255)

    return gesture_name, confidence, status_text, color

def save_sample_to_csv(class_index, features):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([class_index] + features)

def handle_data_collection(key, merged_features, class_index, current_label_tuple):
    # 1. Auto-record for multi-hand gestures
    required_hands = current_label_tuple[1]
    
    try:
        if required_hands > 1:
            if merged_features is None:
                raise ValueError("No hands detected")
            else:
                print("2 or more handed gesture detected, recording automatically")
                save_sample_to_csv(class_index, merged_features)
    except Exception as e:
        print(f"Waiting for hand... (Error: {e})")

    # 2. Manual record
    if key == ord('c') and merged_features is not None:
        save_sample_to_csv(class_index, merged_features)
        print(f"Sample saved for {current_label_tuple}")

# ==============================
# MAIN LOOP
# ==============================
def main():
    # Setup
    model = load_gesture_model()
    initialize_csv()
    
    class_index = 0
    active_effect = None
    effect_timer = 0

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)), # wrap in a str cuz pathlib creates a special Path object (specifically a PosixPath on Mac). MediaPipe tries to .encode() it like a string, which fails.
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.9
    )

    cap = cv2.VideoCapture(0)
    
    with HandLandmarker.create_from_options(options) as landmarker:
        timestamp_ms = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Preparation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            merged_features = None
            gesture_name = ("None", 0)
            confidence = 0.0
            status_text = ""
            color = (0,0,0)

            # 2. Processing (if hands detected)
            if result.hand_landmarks:
                merged_features = merge_two_hands(result)
                
                # Inference
                gesture_name, confidence, status_text, color = perform_inference(model, merged_features)

                # Visualize Hands
                for hand in result.hand_landmarks:
                    
                    draw_hand(frame, hand)

            # 3. Apply Effects (State Update)
            frame, active_effect, effect_timer = apply_visual_effects(
                frame, gesture_name, confidence, active_effect, effect_timer
            )

            # 4. UI Drawing (only if we have inference results or just general overlay)
            draw_ui_overlay(frame, status_text, color, confidence, CLASS_NAMES[class_index], class_index)
            if merged_features is None and result.hand_landmarks:
                 # Fallback if inference didn't run but hands exist (rare)
                 pass
            elif not result.hand_landmarks:
                 # Clear status if no hands
                 pass

            # 5. Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC
                break
            elif key == 9: # TAB
                class_index = (class_index + 1) % len(CLASS_NAMES)
                print(CLASS_NAMES[class_index])

            # 6. Data Collection Logic
            if merged_features is not None:
                handle_data_collection(key, merged_features, class_index, CLASS_NAMES[class_index])
            elif CLASS_NAMES[class_index][1] > 1:
                # Show waiting text if we expect hands but don't see them
                cv2.putText(frame, "WAITING...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 7. Render
            cv2.imshow("Feature Collection Screen", frame)
            timestamp_ms += int(1000 / fps)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()