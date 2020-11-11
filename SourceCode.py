import cv2
import face_recognition as fr
import os
import numpy as np


path = 'img'
imagesName = []
images = []
encodings = []
l = os.listdir(path)

for i in l:
    images.append(fr.load_image_file(f'{path}/{i}'))
    imagesName.append(i.split('.')[0])


def find_encodings(images):
    endcodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode_img = fr.face_encodings(img)[0]
        encodings.append(encode_img)

    return encodings


known_faces_encodings = find_encodings(images)


camera = cv2.VideoCapture(0)
while True:
    success, img = camera.read()
    im = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(img)
    encodings = fr.face_encodings(img)

    for enc, face_loc in zip(encodings, face_locations):
        matches = fr.compare_faces(known_faces_encodings, enc)
        faceDis = fr.face_distance(known_faces_encodings, enc)
        matchIndex = np.argmin(faceDis)

        y1, x2, y2, x1 = face_loc
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if matches[matchIndex]:
            cv2.rectangle(im, (x1, y2 + 35), (x2, y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(im, imagesName[matchIndex], (x1 + 5, y2 + 22), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('x', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
