import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define the new window size
new_width = 1280
new_height = 720

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Resize the frame
    frame = cv2.resize(frame, (new_width, new_height))

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Display finger names at the tip landmarks
            for landmark in [mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                             mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
                             mp_hands.HandLandmark.RING_FINGER_TIP, 
                             mp_hands.HandLandmark.PINKY_TIP, 
                             mp_hands.HandLandmark.THUMB_TIP]:
                # Get the specific landmark
                finger_point = hand_landmarks.landmark[landmark]

                # Convert landmark coordinates to pixel values
                height, width, _ = frame.shape
                cx, cy = int(finger_point.x * width), int(finger_point.y * height)

                # Map landmark to finger name
                finger_name = {
                    mp_hands.HandLandmark.INDEX_FINGER_TIP: "Index finger",
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP: "Middle finger",
                    mp_hands.HandLandmark.RING_FINGER_TIP: "Ring finger",
                    mp_hands.HandLandmark.PINKY_TIP: "Pinky finger",
                    mp_hands.HandLandmark.THUMB_TIP: "Thumb"
                }.get(landmark, "Unknown")

                # Display finger names in black color
                cv2.putText(frame, finger_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Finger Names', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
