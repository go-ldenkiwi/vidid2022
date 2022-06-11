import cv2
import timeit
import time 
import datetime

from cv2 import CAP_FIREWARE


xml = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture('image/sample3.mp4')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
fps = cap.get(cv2.CAP_PROP_FPS) 

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
now = datetime.datetime.now().strftime("%d_%H-%M-%S")
out = cv2.VideoWriter(str(now) + ".avi", fourcc, fps, (int(width), int(height)))
while True:

    start_t = timeit.default_timer()
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    results = xml.detectMultiScale(gray, 1.15, 5)
    
    terminate_t = timeit.default_timer()
    FPS = 'fps' + str(int(1./(terminate_t - start_t )))

    if len(results):
        for (x,y,w,h) in results:
            face_img = img[y:y+h, x:x+w] 
            face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04) 
            face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)
            img[y:y+h, x:x+w] = face_img 
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2) 

    if ret == False:
        break;

    out.write(img)
    cv2.imshow('facenet',img)


    if cv2.waitKey(1) > 0: 
        break


cap.release()
out.release()
cv2.destroyAllWindows()