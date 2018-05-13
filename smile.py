import keras
from keras.models import model_from_json
model = model_from_json(open('model.json').read())
model.load_weights('weights.h5')


import numpy as np
def print_indicator(data, model, class_names, bar_width=50):
    probabilities = model.predict(np.array([data]))[0]
    print(probabilities)
    left_count = int(probabilities[1] * bar_width)
    right_count = bar_width - left_count
    left_side = '-' * left_count
    right_side = '-' * right_count
    print class_names[0], left_side + '###' + right_side, class_names[1]


class_names = ['Neutral', 'Smiling']




from IPython.display import clear_output
from utils import crop_and_resize
import cv2

video_capture = cv2.VideoCapture(0)
try:
    while True:
        ret, frame = video_capture.read()
        cv2.imshow('webcam', frame)
        cv2.waitKey(1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        data = crop_and_resize(gray, 32, zoom=0.6)
        data = data[:, :, np.newaxis]

        data = data.astype(np.float) / 255.
        print_indicator(data, model, class_names)
        clear_output(wait=True)
except KeyboardInterrupt:
    pass
video_capture.release()

