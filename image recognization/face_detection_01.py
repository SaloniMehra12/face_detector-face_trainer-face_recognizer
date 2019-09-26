import cv2

img=cv2.imread("saloni.jpeg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.20,minNeighbors=6)
for x,y,w,h in faces:
    img=cv2.rectangle(img,(x,y),(x+w,y+h),(225,225,225),3)
resized_img=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

cv2.imshow("sample",resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()