import cv2
import time

video=cv2.VideoCapture(0)
id=input("enter student id")
sample_no=0
while True:

    check,frame=video.read()
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.20, minNeighbors=6)

    for x, y, w, h in faces:
        sample_no+=1
        cv2.imwrite("dataSet/Student."+str(id)+"."+str(sample_no)+".jpg",gray_img[y:y+h,x:x+w])
        img = cv2.rectangle(gray_img, (x, y), (x + w, y + h), (225, 225, 225), 3)
        cv2.waitKey(100)
    resized_img = cv2.resize(gray_img, (int(gray_img.shape[1]), int(gray_img.shape[0])))

    cv2.imshow("continuous video",resized_img)
    key=cv2.waitKey(1)
    if sample_no > 20:
        break
        """ if key== ord('l'):#press l to close window
            break"""

print(sample_no)
video.release()
cv2.destroyAllWindows()