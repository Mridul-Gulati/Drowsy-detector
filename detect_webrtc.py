import dlib
from imutils import face_utils
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from pathlib import Path

# Assuming your file is in the same directory as your script
file_path = Path('shape_predictor_68_face_landmarks.dat')

# Convert the Path object to a string
model_path_str = str(file_path.absolute())

# Now use the string representation when creating the shape predictor

status_placeholder = st.empty()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path_str)

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

class VideoProcessor:
    def __init__(self):
        self.sleep = 0
        self.drowsy = 0
        self.active = 0
        self.status = "Active :)"
        self.color = (0, 0, 255)

    def recv(self, frame):
        frm = frame.to_ndarray(format="yuv420p")
        frm = cv2.cvtColor(frm, cv2.COLOR_YUV2BGR_I420)
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frm, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38],
                                landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43],
                                landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if left_blink == 0 or right_blink == 0:
                self.sleep += 1
                self.drowsy = 0
                self.active = 0
                if self.sleep > 4:
                    print("SLEEPING !!!")
                    self.status = "SLEEPING !!!"
                    self.color = (0, 0, 255)

            elif 0.21 < left_blink == 1 or right_blink == 1:
                self.sleep = 0
                self.active = 0
                self.drowsy += 1
                if self.drowsy > 4:
                    print("Drowsy !")
                    self.status = "Drowsy !"
                    self.color = (255, 0, 0)
            else:
                self.drowsy = 0
                self.sleep = 0
                self.active += 1
                if self.active > 3:
                    print("Active :)")
                    self.status = "Active :)"
                    self.color = (0, 255, 0)
        cv2.putText(frm, self.status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)
        return av.VideoFrame.from_ndarray(frm, format="bgr24")
    

webrtc_streamer(key="example", video_processor_factory=VideoProcessor, async_processing=True)
