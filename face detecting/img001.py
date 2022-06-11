import cv2

xml = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
img = cv2.imread('image/md.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = xml.detectMultiScale(gray, 1.05, 5)

for (x,y,w,h) in faces:

	cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
	gray = gray[y:y+h, x:x+w]
	color = img[y:y+h, x:x+w]


cv2.imshow('img',img)

cv2.waitKey(0)

cv2.destroyAllWindows()

