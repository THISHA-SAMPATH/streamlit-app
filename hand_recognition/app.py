import streamlit as st
import cv2
import numpy as np

st.title("Basic Hand Detection using OpenCV")

run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

# Define a simple skin color range in HSV
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Extract skin color image
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Cleaning mask using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=4)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        FRAME_WINDOW.image(res, channels="BGR")

    cap.release()
