import random

import cv2
import numpy as np
import tensorflow
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils.np_utils import to_categorical

import model
from tensorflow.keras.utils import normalize
from datasets import one_hot_encoding, splitDatasets, lists, fw
#from train import history

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print(img.shape,'gray')
    img = cv2.equalizeHist(img)
    #print(img.shape,'equalize')
    img = img/255
    #print(img.shape)
    return img

def prediction():
    X, y = one_hot_encoding()
    x_train, x_valid, x_test, y_train, y_valid, y_test = splitDatasets(X, y)
    #y_train_list, y_test_list, y_valid_list = lists()
    model=tensorflow.keras.models.load_model('finalized_model.h5')
    model.load_weights(fw)
    '''''
    cnn_scores = model.evaluate(x_test, y_test_list)
    print("CNN. Scores: \n", (cnn_scores))
    print("First digit. Accuracy: %.2f%%" % (cnn_scores[6] * 100))
    print("Second digit. Accuracy: %.2f%%" % (cnn_scores[7] * 100))
    print("Third digit. Accuracy: %.2f%%" % (cnn_scores[8] * 100))
    print("Fourth digit. Accuracy: %.2f%%" % (cnn_scores[9] * 100))
    print("Fifth digit. Accuracy: %.2f%%" % (cnn_scores[10] * 100))
    avg_accuracy = sum([cnn_scores[i] for i in range(6, 11)]) / 5
    print("CNN Model. Average Accuracy: %.2f%%" % (avg_accuracy * 100))
    plt.figure(figsize=(11, 5));
    k = 10
    keys = list(history.history.keys())[17:]
    print(keys)
    plt.plot(history.history[keys[0]][k:], label='First digit')
    plt.plot(history.history[keys[1]][k:], label='Second digit')
    plt.plot(history.history[keys[2]][k:], label='Third digit')
    plt.plot(history.history[keys[3]][k:], label='Fourth digit')
    plt.plot(history.history[keys[4]][k:], label='Fifth digit')
    plt.legend();
    plt.title('Accuracy');
    plt.savefig('Accuracy.png')
    '''''

   # print(y_train[5268])
    #print(np.argmax(y_train[5268], axis=1))
    print(len(x_test), 'x_test')

    test_pred = model.predict(x_test)
    print(np.shape(test_pred))
    test_pred = np.transpose(test_pred, [1, 0, 2])

    prediction = []
    for i in range(0, len(test_pred)):
        lable = []
        for j in range(0, 5):
            real_label = np.argmax(test_pred[i][j])
            lable.append(real_label)
        prediction.append(lable)
    np.asarray(prediction)
    print(len(prediction))

    print(np.argmax(y_test[1052], axis=1))

    test_label = [np.argmax(y_test[i], axis=1) for i in range(0, len(y_test))]


    img = x_test[1000]
    cv2.imshow('rand img', img)
    cv2.waitKey(0)
    print(prediction[1000], 'predicted label')
    print(test_label[1000], 'real label')

    for i in range(10):
        rand = random.randint(0, 3200)
        comparison = test_label[rand] == prediction[rand]
        equal_arrays = comparison.all()
        print(equal_arrays)
    lables = [i for i in range(10)]
    test_label = np.array(test_label)
    prediction = np.array(prediction)
    non_None = np.where(test_label.flatten() != 10)
    matrix = confusion_matrix(test_label.flatten()[non_None], prediction.flatten()[non_None], labels=lables)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix, annot=True, cmap='PuRd', fmt='d', ax=ax)
    plt.title('Confusion Matrix for test dataset')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    #plt.savefig('CM_test.png')
    '''''
            for i in range(0, len(test_pred)):
                lable1 = []
                for j in range(0, 5):
                    real_label1 = np.argmax(realTime_pred[i][j])
                    lable1.append(real_label1)
                prediction1.append(lable1)
            np.asarray(prediction1)
            #print(len(prediction1))
            print(prediction1)
            '''''
        #pic=cv2.imread('C:/Users/shahr/Downloads/x-5279.jpg',0)
        #print(pic.shape)
        #pic=cv2.resize(pic,(32,32))
        #print(pic.shape)
        #img=np.expand_dims(pic,axis=2).astype(np.float32)
        #print(img.shape)
        #img = preProcessing(pic)
        #greyscale = np.dot(pic, [0.2989, 0.5870, 0.1140]).astype(np.float32)
        #greyscale=np.expand_dims(greyscale,axis=2)

    return matrix
m1=prediction()
def realTime_prediction():

    model = tensorflow.keras.models.load_model('finalized_model.h5')
    model.load_weights(fw)
    '''''
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        # print(frame.shape)
        img = preProcessing(frame)
        cv2.imshow('frame', img)
        prediction1 = []
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, axis=2).astype(np.float32)
    # print(img.shape)
    img = np.expand_dims(img, 0)
    # print(img.shape)
    realTime_pred = model.predict(img)
    # print(np.shape(realTime_pred))
    # print(realTime_pred)
    realTime_pred = np.transpose(realTime_pred, [1, 0, 2])
    #print(np.argmax(realTime_pred))
    for i in range(0, len(realTime_pred)):
        lable1 = []
        for j in range(0, 5):
            real_label1 = np.argmax(realTime_pred[i][j])
            lable1.append(real_label1)
        prediction1.append(lable1)
    np.asarray(prediction1)
    # print(len(prediction1))
    print(prediction1)
    '''''
    pic=cv2.imread('C:/Users/shahr/Downloads/974.jpg')
    print(pic.shape)
    pic=cv2.resize(pic,(32,32))
    print(pic.shape)
    img=np.expand_dims(pic,axis=2).astype(np.float32)
    print(img.shape)
    img = preProcessing(pic)
    img = cv2.resize(img, (32, 32))
    img = np.expand_dims(img, axis=2).astype(np.float32)
    # print(img.shape)
    img = np.expand_dims(img, 0)
    # print(img.shape)
    realTime_pred = model.predict(img)
    realTime_pred = np.transpose(realTime_pred, [1, 0, 2])
    # print(np.argmax(realTime_pred))
    prediction1=[]
    for i in range(0, len(realTime_pred)):
        lable1 =str()
        for j in range(0, 5):
            real_label1 = np.argmax(realTime_pred[i][j])
            if (real_label1!=10):
                lable1=lable1+str(real_label1)

        #prediction1.append(lable1)
    #np.asarray(prediction1)
    # print(len(prediction1))
    print(lable1)

    return realTime_pred
realTime=realTime_prediction()
