import cv2
import mediapipe as mp
import streamlit as st
import numpy as np

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Gesture Classifier
def classify_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    finger_tips = [8, 12, 16, 20]
    finger_fold_status = []

    for tip in finger_tips:
        if landmarks[tip].y > landmarks[tip - 2].y:
            finger_fold_status.append(True)
        else:
            finger_fold_status.append(False)

    if all(finger_fold_status):
        return "Fist ✊"
    elif not any(finger_fold_status):
        return "Open Palm 🖐️"
    elif not finger_fold_status[0] and all(finger_fold_status[1:]):
        return "Pointing 👉"
    elif thumb_tip.y < index_tip.y and all(f.y > thumb_tip.y for f in [middle_tip, ring_tip, pinky_tip]):
        return "Thumbs Up 👍"
    elif index_tip.y < middle_tip.y < ring_tip.y:
        return "Peace ✌️"
    else:
        return "Unknown Gesture"

# Streamlit App
st.title("✋ Real-Time Hand Gesture Detection")
run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = "No Hand Detected"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                gesture = classify_gesture(landmarks)

        # Show gesture label
        cv2.putText(frame, f'Gesture: {gesture}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    else:
        st.warning("Camera not found or not accessible.")

    cap.release()
