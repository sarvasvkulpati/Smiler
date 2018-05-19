import keras
import time
import line_profiler
from keras.models import model_from_json
model = model_from_json(open('model.json').read())
model.load_weights('weights.h5')



import numpy as np


def print_indicator(data, model, class_names, bar_width=50):
    probabilities = model.predict(np.array([data]))[0]
    #print(probabilities)
    left_count = int(probabilities[1] * bar_width)
    right_count = bar_width - left_count
    left_side = '-' * left_count
    right_side = '-' * right_count
    #print class_names[0], left_side + '###' + right_side, class_names[1]
    return probabilities[1]


class_names = ['Neutral', 'Smiling']




from IPython.display import clear_output
from utils import crop_and_resize
import cv2


classifier = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

smileStart = 0
checkingSmile = False
interval = 5


video_capture = cv2.VideoCapture(0)
try:
    while True:
        currentTime = time.time()
        ret, frame = video_capture.read()
        cv2.imshow('webcam', frame)
        cv2.waitKey(1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
        faces = classifier.detectMultiScale(gray, 1.3, 5)
       
        if len(faces) != 0:
            (x,y,w,h) = faces[0]
                
              
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    
            detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
            
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
           
            detected_face = cv2.resize(detected_face, (32, 32))
            cv2.imshow('face', detected_face)

            data = detected_face[:, :, np.newaxis]
            data = data.astype(np.float) / 255

            prob = print_indicator(data, model, class_names)
            if prob > 0.5 and not checkingSmile:
                print("started smile")
                smileStart = time.time()
                checkingSmile = True
            elif prob > 0.5 and checkingSmile:
                elapsed = currentTime - smileStart 
                print(str(currentTime) + " - " + str(smileStart + interval))
                print("smiled for " + str(elapsed))
                if elapsed > interval:
                    print("you smiled for 5 seconds")
                    checkingSmile = False
                    smileStart = 0
            else:
                checkingSmile = False
                print("not smiling")
    
            
        else:
            print("no face detected")

        
        clear_output(wait=True)
except KeyboardInterrupt:
    pass
video_capture.release()

