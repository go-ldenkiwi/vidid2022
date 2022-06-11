import cv2
import timeit


xml = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

cam = cv2.VideoCapture('image/sample3.mp4')

def videoDetector(cam,xml):
    
    while True:
        
        start_t = timeit.default_timer()
        ret,img = cam.read()
        img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        results = xml.detectMultiScale(gray, 1.15, 5)
        
        for box in results:
            x, y, w, h = box
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), thickness=2)
        
        terminate_t = timeit.default_timer()
        FPS = 'fps' + str(int(1./(terminate_t - start_t )))
        
        cv2.imshow('facenet',img)
        
        if cv2.waitKey(1) > 0: 
            break


videoDetector(cam,xml)
