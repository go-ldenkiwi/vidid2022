import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import datetime

#동영상 비식별화(구분)

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')



def find_faces(img):
    dets = detector(img, 1)
    
    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    
    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)
        
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)
        
    return rects, shapes, shapes_np

def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)

img_paths = {
    'user': 'user_img/user.jpg',
}

descs = []

for name, img_path in img_paths.items(): 
    img = cv2.imread(img_path)
    _, img_shapes, _ = find_faces(img)
    descs.append([name, encode_faces(img, img_shapes)[0]])
    
np.save('user_img/descs.npy', descs)


cap = cv2.VideoCapture('image/sample3.mp4')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
fps = cap.get(cv2.CAP_PROP_FPS) 
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
now = datetime.datetime.now().strftime("%d_%H-%M-%S")
out = cv2.VideoWriter(str(now) + ".avi", fourcc, fps, (int(width), int(height)))

cap.set(3, 640) 
cap.set(4, 480) 
font = cv2.FONT_HERSHEY_SIMPLEX 

while True:
    
    ret, img = cap.read() 

#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rects, shapes, _ = find_faces(img)
    descriptors = encode_faces(img, shapes) 
    
    for i, desc in enumerate(descriptors):
        x = rects[i][0][0] 
        y = rects[i][0][1] 
        w = rects[i][1][1]-rects[i][0][1]
        h = rects[i][1][0]-rects[i][0][0]
        
        
        descs1 = sorted(descs, key=lambda x: np.linalg.norm([desc] - x[1]))
        dist = np.linalg.norm([desc] - descs1[0][1], axis=1)
        
        if dist < 0.6:
            name = descs1[0][0]
        else:    
            name = "unknown"
            mosaic_img = cv2.resize(img[y:y+h, x:x+w], dsize=(0, 0), fx=0.04, fy=0.04) 
            mosaic_img = cv2.resize(mosaic_img, (w, h), interpolation=cv2.INTER_AREA)  
            img[y:y+h, x:x+w] = mosaic_img 

        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2) 
        cv2.putText(img, name, (x+5,y-5), font, 2, (255,255,255), 4)

    out.write(img)        
    cv2.imshow('camera', img)
    
    k = cv2.waitKey(10) & 0xff # 'ESC' 키 누르면 종료
    if k == 27:
        break
        
cap.release()
out.release()
cv2.destroyAllWindows()