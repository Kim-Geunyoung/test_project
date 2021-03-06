import tensorflow.keras
import numpy as np
import cv2
import time
import math

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
global w_time
w_time = 3

global r_time
r_time = int(input("time: "))

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
    
    if (idx == 0):
        time.sleep(1)
        s_time += 1
    elif (idx == 1):
        time.sleep(1)
        o_time += 1  
    elif (idx == 2):
        time.sleep(1)
        p_time += 1
     
    
    if ((s_time % 60) == 0):
        s_time_m += 1
        
    
    if ((p_time % 60) == 0):
        p_time_m += 1
        
        
    if ((o_time % 60) == 0):
        o_time_m += 1
        
        
    if ((p_time % w_time) == 0):
        print("딴짓한 시간이 ", p_time_m, "분 입니다. 집중하세요.")
        
        
    if (s_time_m == r_time):
        print("공부한 시간은 : ", s_time, "입니다.")
        print("자리비운 시간은 : ", o_time, "입니다.")
        print("딴짓한 시간은 : ", p_time, "입니다.")
        sum = s_time + p_time + o_time
        print("금일 학습에 소모된 총 시간은 : ", sum, "입니다.")
        break
    
    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        print("공부한 시간은 : ", s_time, "입니다.")
        print("자리비운 시간은 : ", o_time, "입니다.")
        print("딴짓한 시간은 : ", p_time, "입니다.")
        sum = s_time + p_time + o_time
        print("금일 학습에 소모된 총 시간은 : ", sum, "입니다.")
        break
    
    
    # 여기에 변경된 코딩 정보 작성 ex) 1초 딜레이로 설정해둠
    # Warning message 추가