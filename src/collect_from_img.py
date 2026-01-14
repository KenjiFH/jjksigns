import mediapipe as mp
import cv2

# Aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision

# Paths
MODEL_PATH = "/Users/kenjifahselt/Desktop/proj/jjksigns/models/hand_landmarker.task"
IMAGE_PATH = "/Users/kenjifahselt/Desktop/proj/jjksigns/data/Hand,_fingers_-_back.jpg"

# Load image
mp_image = mp.Image.create_from_file(IMAGE_PATH)

# Create landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)

with HandLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_image)

# Convert MediaPipe image → NumPy
image_np = mp_image.numpy_view()
h, w, _ = image_np.shape
annotated = image_np.copy()

# Hand skeleton connections (MediaPipe standard)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (0,5),(5,6),(6,7),(7,8),        # Index
    (0,9),(9,10),(10,11),(11,12),   # Middle
    (0,13),(13,14),(14,15),(15,16), # Ring
    (0,17),(17,18),(18,19),(19,20)  # Pinky
]

# Draw landmarks
if result.hand_landmarks:
    for hand in result.hand_landmarks:
        # Draw points
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

        # Draw connections
        for start, end in HAND_CONNECTIONS:
            x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
            x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
            cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
else:
    print("No hands detected.")

# Show result
cv2.imshow("Hand Landmarks", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

    
