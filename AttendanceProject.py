import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImageAtt'
images = []
classname=[]
mylsit=os.listdir(path)
print(mylsit)
for cls in mylsit:
    curImage=cv2.imread(f'{path}/{cls}')
    images.append(curImage)
    classname.append(os.path.splitext(cls)[0])
print(classname)

def findencod(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markatt(name):
    with open("att.csv","r+") as f:
        mydata=f.readlines()
        # print(mydata)
        namelist=[]
        for line in mydata:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dt=now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt}')


# markatt("a")
encodeListknown=findencod(images)
print('encoding complete')

cap=cv2.VideoCapture(0)

while True:
    succ,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs,faceCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches=face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListknown,encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name=classname[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
            markatt(name)

    cv2.imshow("webcam",img)
    cv2.waitKey(1)

# faceloc = face_recognition.face_locations(imgel)[0]
# encodeElon = face_recognition.face_encodings(imgel)[0]
# cv2.rectangle(imgel,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)
#
# faceloctest = face_recognition.face_locations(imgte)[0]
# encodeTest = face_recognition.face_encodings(imgte)[0]
# cv2.rectangle(imgte,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)
#
# results = face_recognition.compare_faces([encodeElon],encodeTest)
# faceDis = face_recognition.face_distance([encodeElon],encodeTest)
# print(results,faceDis)