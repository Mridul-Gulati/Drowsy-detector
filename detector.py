import cv2
import numpy as np
import dlib
from imutils import face_utils
import streamlit as st
import base64

# Initializing the camera and taking the instance
cap = cv2.VideoCapture(1)

st.title("Drowsiness Detection App")
frame_placeholder = st.empty()
start_button = st.button("Start")
stop_button = st.button("Stop")
stream = False
if start_button:
    stream = True
if stop_button:
    stream = False
# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Status marking for the current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true" style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if it is blinked
    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

while cap.isOpened() and not stop_button and stream:
    ret, frame = cap.read()
    if not ret or stop_button:
        st.write("The video capture has ended.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Detected face in faces array
    face_frame = frame.copy()

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # The numbers are actually the landmarks which will show the eye
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38],
                             landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Now judge what to do for the eye blinks
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 4:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                autoplay_audio("mixkit-rooster-crowing-in-the-morning-2462.mp3")

        elif 0.21 < left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 4:
                status = "Drowsy !"
                color = (0, 0, 255)
                autoplay_audio("mixkit-street-public-alarm-997.mp3")
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 3:
                status = "Active :)"
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(68):
            x, y = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    # Display the frame in Streamlit
    frame_placeholder.image(frame, channels="BGR")

    # Show the result of the detector in a new window
    #cv2.imshow("Result of detector", face_frame)

    key = cv2.waitKey(100)
    if key == 27 or stop_button:
        break

cap.release()
cv2.destroyAllWindows()
