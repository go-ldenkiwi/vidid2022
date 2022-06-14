import numpy as np
import cv2
import time 
import datetime
import threading
import multiprocessing

xml = 'haarcascades/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(xml)

cap = cv2.VideoCapture(0) # 웹캠을 카메라로 사용
cap.set(3,640) # 너비
cap.set(4,480) # 높이

fourcc = cv2.VideoWriter_fourcc(*'XVID')
count = 99
record1 = False
record2 = False



while(True):
    if(cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        cap.open("/Image/Star.mp4")

    ret, frame = cap.read() 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    faces = face_cascade.detectMultiScale(gray,1.05,5) 
    key = cv2.waitKey(30) & 0xff
    

    cv2.imshow('result0', frame)

    if key == 24: # ctrl x: 녹화
        print("원본 녹화 시작")
        record1 = True
        video = cv2.VideoWriter(str(now) + "original" + ".avi", fourcc, 7.0, (frame.shape[1], frame.shape[0]))

    elif key == 3: # ctrl c: 녹화
        print("원본 녹화 중지")
        record1 = False
        video.release()

    if record1 == True:
        video.write(frame)


    cv2.putText(frame, text=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), org=(30, 450), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2)
    #날짜, 시간 표시

    if len(faces):
        for (x,y,w,h) in faces:
            face_img = frame[y:y+h, x:x+w] # 인식된 얼굴 이미지
            face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04) # face_img의 높이와 너비를 0.04배
            face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA) # 원래 비율로 확대 이 과정에서 이미시 깨짐
            frame[y:y+h, x:x+w] = face_img # 인식된 얼굴 영역을 face_img로 바꿔줌

    cv2.imshow('result', frame)

    
    if key == 27: # Esc: 종료
        break

    elif key == 24: # ctrl x
        print("녹화 시작")
        record2 = True
        video = cv2.VideoWriter(str(now) + ".avi", fourcc, 7.0, (frame.shape[1], frame.shape[0]))
        
    elif key == 3: # ctrl c
        print("녹화 중지")
        record2 = False
        video.release()
    
    if record2 == True:
        print("Number of faces detected: " + str(len(faces)))
        video.write(frame)


cap.release()
cv2.destroyAllWindows()