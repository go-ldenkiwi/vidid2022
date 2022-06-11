import cv2

xml = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
img = cv2.imread('image/md.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = xml.detectMultiScale(gray, 1.05, 5)

if len(faces):
    for (x,y,w,h) in faces:
        face_img = img[y:y+h, x:x+w] 
        face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04)
        face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA)
        img[y:y+h, x:x+w] = face_img 


cv2.imshow('img',img)
cv2.imwrite('test.jpg',img)

cv2.waitKey(0)

cv2.destroyAllWindows()

