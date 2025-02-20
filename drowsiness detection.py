import cv2
import os
import numpy as np
from pygame import mixer
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize alarm
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades
face = cv2.CascadeClassifier('haar cascade files\\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\\haarcascade_righteye_2splits.xml')

# Load the eye state model
model_path = os.path.join(os.getcwd(), 'models', 'cnnCat2.h5')
model = load_model(model_path)

# Thresholds
YAWN_CONSEC_FRAMES = 15

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect face using Haar cascade
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        left_eye = leye.detectMultiScale(roi_gray)
        right_eye = reye.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in right_eye:
            r_eye = roi_gray[ey:ey+eh, ex:ex+ew]
            r_eye = cv2.resize(r_eye, (24, 24)) / 255.0
            r_eye = np.expand_dims(r_eye.reshape(24, 24, -1), axis=0)
            rpred = np.argmax(model.predict(r_eye), axis=-1)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)

        for (ex, ey, ew, eh) in left_eye:
            l_eye = roi_gray[ey:ey+eh, ex:ex+ew]
            l_eye = cv2.resize(l_eye, (24, 24)) / 255.0
            l_eye = np.expand_dims(l_eye.reshape(24, 24, -1), axis=0)
            lpred = np.argmax(model.predict(l_eye), axis=-1)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)

    # Drowsiness detection logic
    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = max(0, score-1)
        cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(frame, f"Score: {score}", (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    if score > 15:
        cv2.putText(frame, "DROWSY", (width//2-100, height//2), font, 3, (0, 0, 255), 2, cv2.LINE_AA)
        sound.play()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
