import numpy as np
import cv2
import PIL as Image
import joblib
import pytesseract
import os
import pickle
from keras.models import load_model

########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.65 # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 1
#####################################

#### CREATE CAMERA OBJECT
cap = cv2.VideoCapture(cameraNo)
cap.set(3,width)
cap.set(4,height)

#### LOAD THE TRAINNED MODEL
model = load_model('finalized_model_10Epoche.h5')
#pickle_in=open('finalized_model.p','rb')
#model=pickle.load(pickle_in)

#### PREPORCESSING FUNCTION
def preProcessing(img):
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   # img = cv2.equalizeHist(img)
    img = img/255
    return img
'''''
while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image",img)
    #img = img.reshape(1,32,32,1)
    #### PREDICT
    #classIndex = int(model.predict_classes(img))
    #print(classIndex)
    #predictions = model.predict(img)
    #print(predictions)
    #probVal= np.amax(predictions)
    #print(classIndex,probVal)

    #if probVal> threshold:
     #   cv2.putText(imgOriginal,str(classIndex) + "   "+str(probVal),
      #              (50,50),cv2.FONT_HERSHEY_COMPLEX,
       #             1,(0,0,255),1)

    #cv2.imshow("Original Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''''

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    success, org_img = cap.read()
    img=np.asarray(org_img)
    #img=cv2.imread('C:/Users/shahr/Downloads/archive/EXP_10/0/0_5.png')
    #print(frame)
    img=preProcessing(img)
    cv2.imshow('frame',img)
    img = cv2.resize(img, (32, 32))
    img=img.reshape(1,32,32,3)
    #print(img.shape)

    #### PREDICT
    predictions= model.predict(img)
    classIndex= np.argmax(predictions,axis=1)
    classNum= np.round(predictions).astype(int)
    #predict_x = model.predict(img)
    #classes_x = np.argmax(predict_x, axis=-1)
    #classIndex = int(model.predict_classes(img))
    #print(classIndex)
    #print(classIndex)
    #predictions = model.predict(img)
    #print(predictions,'----')
    probVal= np.amax(predictions)
    print(classIndex,probVal,classNum)

    if probVal> threshold:
        #cv2.putText(img,str(classIndex) + "   "+str(probVal),
         #        (50,50),cv2.FONT_HERSHEY_COMPLEX,
          #      1,(0,0,255),1)
        #classNum= np.argwhere(classIndex)
        print(str(probVal),'predicted label:',classIndex)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
