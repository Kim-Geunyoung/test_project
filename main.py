from threading import Timer
import tensorflow.keras
import numpy as np
import cv2
import time
import math
import sys

model = tensorflow.keras.models.load_model('keras_model.h5')

cap = cv2.VideoCapture(0)
global s_time
s_time = 1
global p_time
p_time = 1
global o_time
o_time = 1
global s_time_m
s_time_m = 0
global p_time_m
p_time_m = 0
global o_time_m
o_time_m = 0
global r_time

r_time = int(input("timer: "))

size = (224, 224)

classes = ['Study', 'Out', 'Play']

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    h, w, _ = img.shape
    cx = h / 2
    img = img[:, 200:200+img.shape[0]]
    img = cv2.flip(img, 1)

    img_input = cv2.resize(img, size)
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_input = (img_input.astype(np.float32) / 127.0) - 1
    img_input = np.expand_dims(img_input, axis=0)

    prediction = model.predict(img_input)
    idx = np.argmax(prediction)

    cv2.putText(img, text=classes[idx], org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)
   
    # if (idx == 0):
    #     time(s_time)
    # elif (idx == 1):
    #     time(o_time)
    # elif (idx == 2):
    #     time(p_time)
    
    
    if (idx == 0):
        s_time += 1
        time.sleep(1)
    elif (idx == 1):
         o_time += 1
         time.sleep(1)
    elif (idx == 2):
        p_time += 1
        time.sleep(1)
     
     
    
    if ((s_time % 60) == 0):
        s_time_m += 1
        
    if ((o_time % 60) == 0):
        o_time_m += 1
        
    if ((p_time % 60) == 0):
        p_time_m += 1
        
        
    if (s_time_m == r_time):
        print(s_time)
        print(o_time)
        print(p_time)
        sum = s_time + p_time + o_time
        print(sum)
        break
    
    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        print(s_time)
        print(o_time)
        print(p_time)
        sum = s_time + p_time + o_time
        print(sum)
        break
    #