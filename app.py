import numpy as np
import cv2
import tensorflow.keras
from keras.preprocessing import image
import tensorflow as tf


cap= cv2.VideoCapture("test2.mp4")
model = tensorflow.keras.models.load_model("keras_model.h5")
face_cascade = "haarcascade_frontalface.xml"
face_classifier = cv2.CascadeClassifier(face_cascade)
size = (224, 224)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while(True): 
    ret, capture = cap.read()
    capcopy = capture.copy()
    capture = cv2.resize(capture,size)
    img = np.array(capture,dtype=np.float32)
    img = np.expand_dims(img,axis=0)
    img = img/255
    gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.1,4)
    prediction = model.predict(img)

    for (x, y, w, h) in faces:
        
        if(prediction[0][0]>prediction[0][1]) :
            cv2.rectangle(capture, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(capture,"Kuma",(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        else :
            cv2.rectangle(capture, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(capture,"Human",(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
            
    cv2.imshow("image",capture)
    if cv2.waitKey(1) & 0xFF == ord("q") :
        break 

cap.release()
cv2.destroyAllWindows()