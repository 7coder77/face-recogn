import cv2
import numpy as np
import face_recognition

imgel=face_recognition.load_image_file('ImageAtt/Rohit.jpg')
imgel=cv2.cvtColor(imgel,cv2.COLOR_BGR2RGB)
imgte=face_recognition.load_image_file('img/Rohit-Sharma-test.png')
imgte=cv2.cvtColor(imgte,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgel)[0]
encodeElon = face_recognition.face_encodings(imgel)[0]
cv2.rectangle(imgel,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest = face_recognition.face_locations(imgte)[0]
encodeTest = face_recognition.face_encodings(imgte)[0]
cv2.rectangle(imgte,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)


cv2.putText(imgte,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
cv2.imshow('musk' , imgel)
cv2.imshow('musk test' , imgte)
cv2.waitKey(0)