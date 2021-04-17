import numpy as np
import cv2
import random
from tensorflow import keras

# Loading the trained model
model = keras.models.load_model('facial_emotion_classifier_model_new.hdf5')

# Loading haar cascade xml
face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

folders = ['angry', 'happy', 'neutral', 'sad', 'shocked']

def predict(img):
    prediction = model.predict_classes(img.reshape(1,50,50,3))
    text = folders[prediction[0]]
    text = text.upper()
    colour = (0,0,0)  # black
    return text, colour


def draw_stuff_on_frame(frame, colour, text):
    cv2.rectangle(frame, (x,y), (x+w, y+h), colour, 4)  # for bounding box

    # Putting text
    cv2.rectangle(frame,(x,y-30), (x+w, y), colour, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,text,(x+15,y-10), font, 0.5,(255,255,255),2,cv2.LINE_AA) 

################ MAIN ALGORITHM ################
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print('HI')
while True:
    ret, frame = video.read()
    faces = face_cascade.detectMultiScale(frame)
    for (x,y,w,h) in faces:
        roi = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(roi, (50,50))
        normalised_img = resized_img/255
        # Predicting
        text, colour = predict(normalised_img)
        # Visualising the result
        draw_stuff_on_frame(frame, colour, text)
        break

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(25)
    if key == 27: break     # press 'esc' to finish

video.release()
cv2.destroyAllWindows()
