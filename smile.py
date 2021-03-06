import keras
import time
from keras.models import model_from_json
model = model_from_json(open('model.json').read())
model.load_weights('weights.h5')

import numpy as np


def print_indicator(data, model, class_names, bar_width=50):
    probabilities = model.predict(np.array([data]))[0]

    left_count = int(probabilities[1] * bar_width)
    right_count = bar_width - left_count
    left_side = '-' * left_count
    right_side = '-' * right_count
    print class_names[0], left_side + '###' + right_side, class_names[1]



class_names = ['Neutral', 'Smiling']




from utils import crop_and_resize
import cv2

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

smileStart = 0
checkingSmile = False
interval = 5


video_capture = cv2.VideoCapture(0)
try:
    while True:
        currentTime = time.time()
        ret, frame = video_capture.read()
        
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        r = 320.0 /gray.shape[1]
        dim = (320, int(gray.shape[0] * r))
        resized = cv2 .resize(gray, dim, interpolation = cv2.INTER_AREA)
        
        cv2.imshow('webcam', resized)
        cv2.waitKey(1)
        
       
        faces = classifier.detectMultiScale(resized, 1.3, 5)
       

        if len(faces) != 0:
            (x,y,w,h) = faces[0]
                
            cv2.rectangle(resized,(x,y),(x+w,y+h),(255,0,0),2)
                    
            detected_face = resized[int(y):int(y+h), int(x):int(x+w)] #crop detected face
            
            #detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
            detected_face = cv2.resize(detected_face, (32, 32))
            cv2.imshow('face', detected_face)

            data = detected_face[:, :, np.newaxis]
            data = data.astype(np.float) / 255

            print_indicator(data, model, class_names)

        else:
            print("no face detected")

        

except KeyboardInterrupt:
    pass
video_capture.release()

