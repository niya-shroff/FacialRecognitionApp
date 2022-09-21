import cv2
import numpy as np
import face_recognition
import os

path = 'facesTest'
images = []
classNames = []
faces = os.listdir(path)

for cl in faces:
    currImg = cv2.imread(f'{path}/{cl}')
    images.append(currImg)
    classNames.append(os.path.splitext(cl)[0])

images.remove(images[0])
classNames.remove(classNames[0])

encodeList = []
for img in images:
    imgN = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(imgN)[0]
    encodeList.append(encode)

print('Encoding Finished')

cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)

    for encodeFace, faceLocation in zip(encodeCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodeList, encodeFace)
        faceDistance = face_recognition.face_distance(encodeList, encodeFace)
        print(faceDistance)
        matchIndex = np.argmin(faceDistance)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLocation
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
