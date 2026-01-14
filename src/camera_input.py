import mediapipe as mp
import cv2
import math
import numpy as np


import pickle

import csv  
import os   

# ==============================
#This is the main loop to record and reference hand landmarks using mediapipe
# ==============================



# ==============================
# MediaPipe setup
# ==============================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
#our mediapipe model path
MODEL_PATH = "/Users/kenjifahselt/Desktop/proj/jjksigns/models/hand_landmarker.task"


# ==============================
# Feature extraction ( keep SAME normalization for training loop)
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
        coords.append([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])

    # 2. Scale invarience (Reference Length Normalization) normalization to account for different size hands or distance from camera
    ref = coords[9]  # middle finger landmarks
    scale = math.sqrt(ref[0]**2 + ref[1]**2 + ref[2]**2) + 1e-6 # safety epsilon.

    coords = [[c / scale for c in p] for p in coords]

    # 3. Flatten
    feature_vector = []
    for p in coords:
        feature_vector.extend(p)

    return feature_vector




EMPTY_HAND = [0.0] * 63  # one-hand feature size





def wrist_xy(hand):
    return np.array([hand[0].x, hand[0].y])

#takes in a MediaPipe HandLandmarkerResult object or a snapshot of everything detected in one frame
#Gets feature vectors from landmarks of both hand, returns tuple of L and R features as well as wrist distance
#and a tuple of the hand presance
def merge_two_hands(result):
    """
    result: MediaPipe HandLandmarkerResult
    returns: merged feature vector (Left | Right | cross-hand)
    """

    hands = {}
    for idx, hand in enumerate(result.hand_landmarks):

        handedness_info = result.handedness[idx][0]
        handedness = handedness_info.category_name  
       

        # ---- ACCEPT HAND ----
        hands[handedness] = hand
    

        

    # ---- Per-hand features ----
    if "Left" in hands:
        left_feat = hand_to_feature_vector(hands["Left"])
        left_wrist = np.array([
            hands["Left"][0].x,
            hands["Left"][0].y,
            hands["Left"][0].z
        ])
    else:
        left_feat = EMPTY_HAND
        left_wrist = None

    if "Right" in hands:
        right_feat = hand_to_feature_vector(hands["Right"])
        right_wrist = np.array([
            hands["Right"][0].x,
            hands["Right"][0].y,
            hands["Right"][0].z
        ])
    else:
        right_feat = EMPTY_HAND
        right_wrist = None

    # ---- Cross-hand features ----
    if left_wrist is not None and right_wrist is not None:
        wrist_dist = np.linalg.norm(left_wrist - right_wrist)
    else:
        wrist_dist = 0.0

    # Optional presence flags (recommended)
    presence = [
        int(left_wrist is not None),
        int(right_wrist is not None)
    ]

    return left_feat + right_feat + [wrist_dist] + presence

# ==============================
# Drawing helpers
# ==============================
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def draw_hand(frame, hand_landmarks):
    h, w, _ = frame.shape

    xs = []
    ys = []

    # Draw landmarks
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        xs.append(cx)
        ys.append(cy)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # Draw skeleton
    for start, end in HAND_CONNECTIONS:
        x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
        x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)



# ==============================
# Func to train 2 handed gestures - user cant press c
# ==============================
def train_two_handed(class_idx):
    pass

# ==============================
# Load Model
# ==============================

# 1. Load the trained model
# Ensure 'gesture_model.pkl' is in the correct path!
try:
    with open('/Users/kenjifahselt/Desktop/proj/jjksigns/notebooks/gesture_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Warning: Model not found. Running in Data Collection mode only.")
    model = None




# ==============================
# CONFIGURATION: this is where the gesture labels are stored
# ==============================
# Define your class list here
CLASS_NAMES = [
    ("ThumbsUp", 1),      # Must see exactly 1 hand
    ("Peace", 1),         # Must see exactly 1 hand
    ("Gojo_Void", 1), # Must see exactly 2 hands
    ("Sukuna_Shrine", 2),  # Must see exactly 2 hands
    ("Cinema", 2)
] #it woudl probably be best to make these mapped enums [name, num_hands]


class_index = 0
TARGET_LABEL = CLASS_NAMES[class_index]
CSV_FILE = '/Users/kenjifahselt/Desktop/proj/jjksigns/models/keypoint_classifier/keypoints.csv'

# Create the CSV header if the file doesn't exist yet
if not os.path.exists(CSV_FILE):
    # 129 features = 63 (Left) + 63 (Right) + 1 (Dist) + 2 (Presence)
    header = [f"f_{i}" for i in range(129)] + ["label"]
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)



# --- EFFECT STATE MANAGEMENT ---
active_effect = None  # Can be "GOJO", "SUKUNA", or None
effect_timer = 0      # Counts down frames (e.g., 60 frames = ~2 seconds)

# ==============================
# Main webcam loop
# ==============================
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence = 0.9
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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
            
        )

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        

        if result.hand_landmarks:
            # ---- MERGED TWO-HAND FEATURE VECTOR ----
            merged_feature_vector = merge_two_hands(result)
            #print( "Feature dim:", len(merged_feature_vector),  "Presence:", merged_feature_vector[-2:])
              # debug: dim should be 129
              # 21 landmarks × 3 coordinates (x,y,z) = 63 values 2 * 63 = 126 
              # 129 features = 63 (Left) + 63 (Right) + 1 (Dist) + 2 (Presence)
              
              # https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python


            # ==============================
            #  Model Inference + decision logic goes here
            # ==============================
            # ---- Model inference ----

            if model is not None:
            
            # ---------------------------------------------------------
            # START INFERENCE BLOCK
            # ---------------------------------------------------------
            
                # 1. Preprocess: Reshape the list into a 2D array (1 sample, 129 features)
                input_data = np.array(merged_feature_vector).reshape(1, -1)

                # 2. Predict: Get probability distribution (Softmax-like)
                # Example Output: [[0.05, 0.90, 0.02, 0.03]]
                probs = model.predict_proba(input_data)[0]

                # 3. Interpret: Find the class with the highest score
                max_idx = np.argmax(probs)
                confidence = probs[max_idx]
                gesture_name = CLASS_NAMES[max_idx]

                # 4. Filter: Only accept if confidence is high enough
                # (This helps filter out transitions/garbage movements)
                CONFIDENCE_THRESHOLD = 0.80  

                if confidence > CONFIDENCE_THRESHOLD:
                    # Valid Detection
                    status_text = f"{gesture_name}: {confidence:.2f}"
                    color = (0, 255, 0) # Green
                else:
                    # Low Confidence (Uncertain)
                    status_text = f"Uncertain ({gesture_name}? {confidence:.2f})"
                    color = (0, 165, 255) # Orange
            # ---------------------------------------------------------
            # end INFERENCE BLOCK
            # ---------------------------------------------------------
            # ... (After getting gesture_name and confidence) ...
    
            # ---------------------------------------------------------
            # 1. TRIGGER LOGIC (Set the Effect)
            # ---------------------------------------------------------
            if confidence  > 0.90:
                
                if gesture_name[0] == "Gojo_Void":
                    active_effect = "GOJO"
                    effect_timer = 30  # Effect lasts 60 frames (approx 2 seconds)
                elif gesture_name[0] == "Sukuna_Shrine":
                    active_effect = "SUKUNA"
                    effect_timer = 30

                # ---------------------------------------------------------
                # RENDER LOGIC (Draw the Flair)
                # ---------------------------------------------------------
                if effect_timer > 0:
                    # Create a colored overlay layer same size as the frame
                    overlay = np.zeros_like(frame)
                    
                    # Get screen dimensions for positioning
                    h, w, _ = frame.shape
                    
                    if active_effect == "GOJO":
                        # --- 1. The Full Screen Purple Tint ---
                        overlay[:] = (255, 20, 180) 
                        frame = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)
                        
                        # --- 2. Right-Aligned Text ---
                        text = "DOMAIN EXPANSION: UNLIMITED VOID"
                        font = cv2.FONT_HERSHEY_DUPLEX
                        scale = 1.0
                        thickness = 2
                        
                        # Calculate text size to determine X position
                        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
                        
                        # X = Width - Text_Width - 20px Padding
                        x_pos = w - text_w - 20 
                        y_pos = 100
                        
                        cv2.putText(frame, text, (x_pos, y_pos), font, scale, (255, 255, 255), thickness)

                    elif active_effect == "SUKUNA":
                        # --- 1. The Full Screen Red Tint ---
                        overlay[:] = (0, 0, 200)
                        frame = cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)
                        
                        # --- 2. Right-Aligned Text ---
                        text = "DOMAIN EXPANSION: MALEVOLENT SHRINE"
                        font = cv2.FONT_HERSHEY_DUPLEX
                        scale = 1.0
                        thickness = 2
                        
                        # Calculate text size
                        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
                        
                        # X = Width - Text_Width - 20px Padding
                        x_pos = w - text_w - 20
                        y_pos = 100
                        
                        cv2.putText(frame, text, (x_pos, y_pos), font, scale, (0, 0, 255), thickness)

                    # Count down
                    #TODO fix false pos
                    print(f" gesture {gesture_name[0]} count: {effect_timer} for effect {active_effect}")
                    effect_timer -= 1
                    
                else:
                    # Reset state when timer hits 0
                    active_effect = None

                # ... (Now call cv2.imshow) ...

             # ---------------------------------------------------------
            #  inference viz block
            # ---------------------------------------------------------
                # 5. Visualize: Draw the result on screen
                # Main label
               # Get the height of the frame so we can position relative to bottom
                h, w, _ = frame.shape 
                
            
                base_y = h - 400

        
                cv2.putText(frame, status_text, (20, base_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                
                # Optional: Show a "Confidence Bar" below the text
               
                bar_y_start = base_y + 10
                bar_y_end = base_y + 30
                
                bar_width = int(confidence * 200)
                
                # Filled Bar
                cv2.rectangle(frame, (20, bar_y_start), (20 + bar_width, bar_y_end), color, -1)
                # Border outline
                cv2.rectangle(frame, (20, bar_y_start), (220, bar_y_end), (255, 255, 255), 2)
            # ---------------------------------------------------------
            # end inference viz block
            # ---------------------------------------------------------
               
                
               
          


            # ---- HAND LANDMARK VISUALIZATION ----
            for hand in result.hand_landmarks:
                draw_hand(frame, hand)

        cv2.putText(frame, f"Current Label: {TARGET_LABEL} at index: {class_index} (Press TAB to switch)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Hold 'c' to record", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        


       
     

        # ==============================
        #  INPUT HANDLING & SAVING
        # ==============================

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to quit
            break

        # Press TAB (ascii 9) to cycle through labels
        elif key == 9: 
            
            class_index = (class_index + 1) % len(CLASS_NAMES)
            TARGET_LABEL = CLASS_NAMES[class_index]
            print(CLASS_NAMES[class_index])

         # ---- TRAINING LOOP ----
         # 
         #Save sample to models/keypoint_classifier/keypoints.csv
        required_hands = CLASS_NAMES[class_index][1]

        #collection for 2 handed gestures (user cant press c)
        try:
            # Check if we should auto-record
            if required_hands > 1:
                
                # This line forces a crash if the vector is None/Empty
                if merged_feature_vector is None:
                    raise ValueError("No hands detected")
                else:
                    print("2 or more handed gesture detected, recording automatically")
                    
                    # Save logic
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([class_index] + merged_feature_vector)

        except Exception as e:
            # This block runs if ANYTHING goes wrong above
            # It catches the crash and just prints a message instead
            print(f"Waiting for hand... (Error: {e})")
            cv2.putText(frame, "WAITING...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
         

        if key == ord('c') and merged_feature_vector is not None:
            # Append to CSV
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                 # Write: [Label_ID, Feature_1, Feature_2, ... Feature_129]
                writer.writerow([class_index] + merged_feature_vector)
            
            #  feedback that we are recording
            
            print(f"Sample saved for {TARGET_LABEL}")

            
 
        cv2.imshow("Feature Collection Screen", frame)
       
        timestamp_ms += int(1000 / fps)

cap.release()
cv2.destroyAllWindows()

