import mediapipe as mp
import cv2
import math
import numpy as np
import pickle
import csv
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path

# ==============================
# CONFIGURATION & CONSTANTS
# ==============================

# 1. Path Setup
SCRIPT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = SCRIPT_DIR / "models" / "hand_landmarker.task"
PICKLE_MODEL_PATH = SCRIPT_DIR / "notebooks" / "gesture_model.pkl"
CSV_FILE = SCRIPT_DIR / "models" / "keypoint_classifier" / "keypoints.csv"

print(f"Loading model from: {MODEL_PATH}")

# UPDATED CLASS NAMES
CLASS_NAMES = [
    ("ThumbsUp", 1),      
    ("Peace", 1),         
    ("Gojo_Void", 1),     
    ("Sukuna_Shrine", 2), 
    ("Unknown_2_Hand", 2), # New: Junk data for 2 hands (e.g., clapping, crossed arms)
    ("Unknown", 1)         # New: Junk data for 1 hand (e.g., scratching nose, resting)
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
# 1. STRATEGY PATTERN CLASSES
# ==============================

class VisualEffect(ABC):
    """Abstract Strategy: The blueprint for any visual effect."""
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
        """Draw the effect on the frame. Must be implemented by concrete strategies."""
        pass

    def _draw_centered_text(self, frame, text, color):
        """Helper to draw centered text at top right."""
        h, w, _ = frame.shape
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.0
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        
        # Position: Top Right with padding
        x = w - text_w - 20
        y = 100
        cv2.putText(frame, text, (x, y), font, scale, color, thickness)


class GojoEffect(VisualEffect):
    """Concrete Strategy 1: Infinite Void (Purple, Inverted, Stars)"""
    def render(self, frame):
        if not self.is_active:
            return frame
        
        h, w, _ = frame.shape

        # 1. "Information Overload" (Inverted Colors Blend)
        inverted_frame = cv2.bitwise_not(frame)
        frame = cv2.addWeighted(frame, 0.6, inverted_frame, 0.4, 0) 

        # 2. "The Void" Atmosphere (Deep Purple Overlay)
        overlay = np.zeros_like(frame)
        overlay[:] = (255, 50, 100) # Deep Violet (BGR)
        
        # 3. Dynamic Stars
        for _ in range(50):
            star_x = random.randint(0, w)
            star_y = random.randint(0, h)
            star_size = random.randint(1, 3) 
            cv2.circle(overlay, (star_x, star_y), star_size, (255, 255, 255), -1)
            
        # Blend the Starry Overlay
        frame = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)

        # Draw Text
        self._draw_centered_text(frame, "DOMAIN EXPANSION: UNLIMITED VOID", (255, 255, 255))
        
        return frame


class SukunaEffect(VisualEffect):
    """Concrete Strategy 2: Malevolent Shrine (Red, Slashes)"""
    def render(self, frame):
        if not self.is_active:
            return frame

        h, w, _ = frame.shape
        
        # Red Tint Overlay
        overlay = np.zeros_like(frame)
        overlay[:] = (0, 0, 200) # Red (BGR)
        frame = cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)
        
        #Rapid Slashes (Randomized every frame)
        num_slashes = random.randint(5, 10) 
        for _ in range(num_slashes):
            if random.choice([True, False]): 
                # Horizontal-ish
                x1, x2 = 0, w
                y1 = random.randint(0, h)
                y2 = random.randint(0, h)
            else:
                # Vertical-ish
                x1 = random.randint(0, w)
                x2 = random.randint(0, w)
                y1, y2 = 0, h

            thickness = random.randint(1, 4)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness)

        # Draw Text
        self._draw_centered_text(frame, "DOMAIN EXPANSION: MALEVOLENT SHRINE", (0, 0, 255))
        
        return frame


# ==============================
# HELPER FUNCTIONS: Logic & Math
# ==============================
def hand_to_feature_vector(hand):
    if hand is None or len(hand) != 21:
        raise ValueError("Expected 21 hand landmarks")
    wrist = hand[0]
    coords = []
    for lm in hand:
        coords.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    ref = coords[9] 
    scale = math.sqrt(ref[0]**2 + ref[1]**2 + ref[2]**2) + 1e-6 
    coords = [[c / scale for c in p] for p in coords]
    feature_vector = []
    for p in coords:
        feature_vector.extend(p)
    return feature_vector

def merge_two_hands(result):
    hands = {}
    for idx, hand in enumerate(result.hand_landmarks):
        handedness_info = result.handedness[idx][0]
        handedness = handedness_info.category_name  
        hands[handedness] = hand

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

    if left_wrist is not None and right_wrist is not None:
        wrist_dist = np.linalg.norm(left_wrist - right_wrist)
    else:
        wrist_dist = 0.0

    presence = [int(left_wrist is not None), int(right_wrist is not None)]
    return left_feat + right_feat + [wrist_dist] + presence

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
        header = [f"f_{i}" for i in range(129)] + ["label"]
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def draw_hand(frame, hand_landmarks):
    h, w, _ = frame.shape
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    for start, end in HAND_CONNECTIONS:
        x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
        x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

def draw_ui_overlay(frame, status_text, color, confidence, target_label, class_index):
    h, w, _ = frame.shape 
    base_y = h - 400
    cv2.putText(frame, status_text, (20, base_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    bar_y_start = base_y + 10
    bar_y_end = base_y + 30
    bar_width = int(confidence * 200)
    cv2.rectangle(frame, (20, bar_y_start), (20 + bar_width, bar_y_end), color, -1)
    cv2.rectangle(frame, (20, bar_y_start), (220, bar_y_end), (255, 255, 255), 2)
    cv2.putText(frame, f"Training Current Label: {target_label} at index: {class_index} (Press TAB to switch)", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Hold 'c' to record Frame", (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def perform_inference(model, merged_features):
    if model is None:
        return ("None", 0), 0.0, "Model Not Loaded", (100, 100, 100)

    input_data = np.array(merged_features).reshape(1, -1)
    probs = model.predict_proba(input_data)[0]
    max_idx = np.argmax(probs)
    confidence = probs[max_idx]
    
    gesture_name_tuple = CLASS_NAMES[max_idx]
    label_str = gesture_name_tuple[0]

    # --- UNKNOWN CLASS HANDLING ---
    # If the model thinks this is "Unknown" or "Unknown_2_Hand", we force a rejection.
    # We return "None" so the UI handler knows to clear any active effects.
    if label_str == "Unknown" or label_str == "Unknown_2_Hand":
         return ("None", 0), 0.0, "Status: Idle (Unknown)", (200, 200, 200)

    CONFIDENCE_THRESHOLD = 0.90  
    if confidence > CONFIDENCE_THRESHOLD:
        status_text = f"{label_str}: {confidence:.2f}"
        color = (0, 255, 0) 
    else:
        status_text = f"Uncertain ({label_str}? {confidence:.2f})"
        color = (0, 165, 255)

    return gesture_name_tuple, confidence, status_text, color

def save_sample_to_csv(class_index, features):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([class_index] + features)

def handle_data_collection(key, merged_features, class_index, current_label_tuple):
    required_hands = current_label_tuple[1]
    
    # NOTE: We probably don't want to auto-record "Unknown_2_Hand" unintentionally
    # But if you selected it with TAB, you probably DO want to record it.
    
    try:
        # Only auto-record if it's NOT an "Unknown" class (unless you want to spam record junk)
        is_unknown = "Unknown" in current_label_tuple[0]
        
        if required_hands > 1 and not is_unknown:
            if merged_features is None:
                raise ValueError("No hands detected")
            else:
                save_sample_to_csv(class_index, merged_features)
    except Exception as e:
        pass

    if key == ord('c') and merged_features is not None:
        save_sample_to_csv(class_index, merged_features)
        print(f"Sample saved for {current_label_tuple}")

# ==============================
# 2. NEW EFFECTS HANDLER
# ==============================
def handle_effects(frame, gesture_name_tuple, confidence, active_effect_instance, registry):
    gesture_label = gesture_name_tuple[0] 

    # --- A. TRIGGER LOGIC ---
    if confidence > 0.90 and gesture_label in registry:
        active_effect_instance = registry[gesture_label]
        active_effect_instance.start()
    
    # --- B. CANCEL LOGIC ---
    # If hand is lost, confidence drops, OR we see an "Unknown" class (which returns "None" from inference)
    if gesture_label == "None" or confidence < 0.50:
        active_effect_instance = None

    # --- C. RENDER LOGIC ---
    if active_effect_instance and active_effect_instance.is_active:
        active_effect_instance.update()
        frame = active_effect_instance.render(frame)
    else:
        active_effect_instance = None

    return frame, active_effect_instance


# ==============================
# MAIN LOOP
# ==============================
def main():
    model = load_gesture_model()
    initialize_csv()
    class_index = 0
    
    # --- SETUP: Initialize The Strategy Registry ---
    effect_registry = {
        "Gojo_Void": GojoEffect(duration=30),
        "Sukuna_Shrine": SukunaEffect(duration=30)
    }
    
    # State variable now holds the OBJECT, not a string
    current_effect_obj = None 

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)), 
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
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            merged_features = None
            gesture_name = ("None", 0)
            confidence = 0.0
            status_text = ""
            color = (0,0,0)

            if result.hand_landmarks:
                merged_features = merge_two_hands(result)
                gesture_name, confidence, status_text, color = perform_inference(model, merged_features)
                for hand in result.hand_landmarks:
                    draw_hand(frame, hand)

            # --- USE NEW HANDLER ---
            frame, current_effect_obj = handle_effects(
                frame, 
                gesture_name, 
                confidence, 
                current_effect_obj, 
                effect_registry
            )

            draw_ui_overlay(frame, status_text, color, confidence, CLASS_NAMES[class_index], class_index)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            elif key == 9: 
                class_index = (class_index + 1) % len(CLASS_NAMES)
                print(CLASS_NAMES[class_index])

            if merged_features is not None:
                handle_data_collection(key, merged_features, class_index, CLASS_NAMES[class_index])
            elif CLASS_NAMES[class_index][1] > 1:
                cv2.putText(frame, "WAITING...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Feature Collection Screen", frame)
            timestamp_ms += int(1000 / fps)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()