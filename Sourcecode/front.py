import cv2
import os
import numpy as np
from keras.models import model_from_json
import logging as log
import datetime as dt
from time import sleep
from utils.inference import apply_offsets
import glob
import skimage.io
from sklearn.preprocessing import LabelBinarizer

ck_path = 'downloads'
cascPath = "./models/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)


json_file = open('newmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("newmodel.h5")
print("Loaded model from disk")


# Creat the list to store the data and label information
data_x = []
data_y_e = []
data_y_f = []

# Directory containing training images
directory = glob.glob("Webcam_team_dataset/*")

# Read images in all folders and build labels for face and expression simultaneously
for folder in directory:
    files = os.listdir(folder)
    files.sort()
    for filename in files:
        if filename=='.DS_Store':
            continue
        I = skimage.io.imread(os.path.join(folder,filename))
        data_x.append(I.tolist())
        label1= folder.split("/")
        label2= filename.split('_')
        data_y_f.append([label2[0]])
        data_y_e.append([label1[1]])


data_y_f=np.array(data_y_f)
data_y_e=np.array(data_y_e)

mlb1 = LabelBinarizer()
mlb2 = LabelBinarizer()
label1 = mlb1.fit_transform(np.array(data_y_f))
label2 = mlb2.fit_transform(np.array(data_y_e))

print(mlb1.classes_)

video_capture = cv2.VideoCapture(0)
anterior = 0
emotion_offsets = (20, 40)
i=108
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()


    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces



    for face_coordinates in faces:

        x, y, w,h = face_coordinates
        cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 2)
        gray_face=frame[y:y+h,x:x+w]

        try:
            gray_face = cv2.resize(gray_face, (48,48))
        except:
            continue

        #gray_face = np.array(gray_face, dtype=np.float32)
        #gray_face /= 255
        cv2.imwrite('t/Anger/harsh_angewng'+str(i)+'.jpg', gray_face)
        i+=1

        (iden,exp) = loaded_model.predict(gray_face.reshape(1,48,48,3))
        iden_ind = iden.argmax()
        exp_ind = exp.argmax()
        categoryLabel = mlb1.classes_[iden_ind]
        colorLabel = mlb2.classes_[exp_ind]

        categoryText = "Person: {} ({:.2f}%)".format(categoryLabel,
	    iden[0][iden_ind] * 100)
        colorText = "Emo: {} ({:.2f}%)".format(colorLabel,
	    exp[0][exp_ind] * 100)
        cv2.putText(frame, categoryText, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)
        cv2.putText(frame, colorText, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
