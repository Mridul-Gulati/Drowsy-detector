import cv2
import numpy as np
import dlib
from imutils import face_utils
from flask import Flask, render_template, Response
import pygame

app = Flask(__name__,static_folder='static')

predictor_path = 'shape_predictor_68_face_landmarks.dat'

cap = cv2.VideoCapture(0,cv2.CAP_ANY)
# cap = VideoStream(src=0).start()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# pygame.mixer.init()
# alert_sound = pygame.mixer.Sound("static/mixkit-rooster-crowing-in-the-morning-2462.mp3")

# def playAlertSound():
#     alert_sound.play()

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

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

def gen_frames():
    global sleep
    global drowsy
    global active
    global status
    global color
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            face_frame = frame.copy()

            for face in faces:
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

                cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                landmarks = predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                left_blink = blinked(landmarks[36], landmarks[37], landmarks[38],
                                    landmarks[41], landmarks[40], landmarks[39])
                right_blink = blinked(landmarks[42], landmarks[43],
                                    landmarks[44], landmarks[47], landmarks[46], landmarks[45])

                if left_blink == 0 or right_blink == 0:
                    sleep += 1
                    drowsy = 0
                    active = 0
                    if sleep > 8:
                        status = "SLEEPING !!!"
                        color = (255, 0, 0)
                        # playAlertSound()

                elif 0.21 < left_blink == 1 or right_blink == 1:
                    sleep = 0
                    active = 0
                    drowsy += 1
                    if drowsy > 8:
                        status = "Drowsy !"
                        color = (0, 0, 255)
                        # playAlertSound()
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

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__=='__main__':
#     app.run(debug=True)
