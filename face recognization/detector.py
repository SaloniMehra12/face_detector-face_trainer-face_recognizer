import cv2
import time

video=cv2.VideoCapture(0)
a=1
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer/trainingData.yml')
id=0
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)
while True:
    a=a+1
    check,frame=video.read()
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.20, minNeighbors=6)
    for x, y, w, h in faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 225, 225), 3)
        id,conf=rec.predict(gray_img[y:y+h,x:x+w])
        cv2.putText(frame, str(id), (x,y+h), fontface, fontscale, fontcolor,3)
    resized_img = cv2.resize(frame, (int(frame.shape[1]), int(frame.shape[0])))

    cv2.imshow("continuous video",resized_img)
    key=cv2.waitKey(1)
    if key== ord('l'):#press l to close window
        break
print(a)
video.release()
cv2.destroyAllWindows()