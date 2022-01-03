import cv2
import os

path = os.path.dirname(cv2.__file__) + "\\data\\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(path)

videoCapture = cv2.VideoCapture(0)
while(1):
    ret, frame = videoCapture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(30,30)
    )
    for x,y,w,h in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
    
    cv2.imshow("Face", frame)
    if (cv2.waitKey(1) & 0xFF== ord('q')):
        break;