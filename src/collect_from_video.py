import mediapipe as mp
import cv2

# Aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Paths
MODEL_PATH = "/Users/kenjifahselt/Desktop/proj/jjksigns/models/hand_landmarker.task"
VIDEO_PATH_GOJO = "/Users/kenjifahselt/Desktop/proj/jjksigns/data/gojo_domain_clip.mov"
VIDEO_PATH =  "/Users/kenjifahselt/Desktop/proj/jjksigns/data/input_vid.mp4"




# Hand skeleton connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # Thumb
    (0,5),(5,6),(6,7),(7,8),        # Index
    (0,9),(9,10),(10,11),(11,12),   # Middle
    (0,13),(13,14),(14,15),(15,16), # Ring
    (0,17),(17,18),(18,19),(19,20)  # Pinky
]

# Create landmarker (VIDEO mode)
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO
)

cap = cv2.VideoCapture(VIDEO_PATH_GOJO)

with HandLandmarker.create_from_options(options) as landmarker:
    timestamp_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap frame in MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Detect hands
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w, _ = frame.shape

        # Draw landmarks
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                # Points
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                # Connections
                for start, end in HAND_CONNECTIONS:
                    x1, y1 = int(hand[start].x * w), int(hand[start].y * h)
                    x2, y2 = int(hand[end].x * w), int(hand[end].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Hand Landmarks (Video)", frame)

        # Exit on ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

        timestamp_ms += int(1000 / cap.get(cv2.CAP_PROP_FPS))

cap.release()
cv2.destroyAllWindows()
