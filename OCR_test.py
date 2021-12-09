import numpy as np
import cv2
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


#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = img/255
    return img

def capture():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        #success, org_img = cap.read()
        #img=np.asarray(org_img)
        img=cv2.imread('C:/Users/shahr/Downloads/archive/EXP_10/0/0_5.png')
        #print(frame)
        img=preProcessing(img)
        cv2.imshow('frame',img)
        img = cv2.resize(img, (32, 32))
        img=img.reshape(1,32,32,3)
        #print(img.shape)

        #### PREDICT
        predictions= model.predict(img)
        classIndex= np.argmax(predictions)+1
        classNum= np.round(predictions).astype(int)
        if classIndex==10:
            classIndex=0
        probVal= np.amax(predictions)
        #print(classIndex,probVal,classNum)

        if probVal> threshold:
            print(str(probVal),'predicted label:',classIndex)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return probVal

probVal=capture()
