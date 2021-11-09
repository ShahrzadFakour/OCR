import numpy as np
import tensorflow
import os
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from tensorflow import keras
import cv2
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def read_digits():
    # Path to the data directory
    path=r'C:/Users/shahr/Downloads/archive/EXP_10'

    count = 0
    images = []  # LIST CONTAINING ALL THE IMAGES
    classNo = []  # LIST CONTAINING ALL THE CORRESPONDING CLASS ID OF IMAGES
    myList = os.listdir(path)
    print("Total Classes Detected:", len(myList))
    noOfClasses = len(myList)
    print("Importing Classes .......")
    for x in range(0, noOfClasses):
        myPicList = os.listdir(path + "/" + str(x))
        for y in myPicList:
            curImg = cv2.imread(path + "/" + str(x) + "/" + y)
            curImg = cv2.resize(curImg, (32, 32))
            images.append(curImg)
            classNo.append(x)
        print(x, end=" ")

    #### CONVERT TO NUMPY ARRAY
    images = np.array(images)
    classNo = np.array(classNo)
    print(images.shape)
    return images,classNo,noOfClasses
images,classNo,noOfClasses=read_digits()
print(" ")
print("Total Images in Images List = ", len(images))
print("Total IDS in classNo List= ", len(classNo))
print(images.shape)

images,labels,characters=read_digits()

def splitting_data():
    images, classNo, noOfClasses = read_digits()
    testRatio=0.2
    x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=testRatio)
    y_train = to_categorical(y_train, noOfClasses)
    y_test = to_categorical(y_test, noOfClasses)
    y_validation = to_categorical(y_validation, noOfClasses)
    print(x_train.shape)
    print(x_test.shape)
    print(x_validation.shape)
    return x_train, x_test, x_validation, y_train, y_test, y_validation

def create_model():
    model=Sequential()
    model.add(Conv2D(32,(3,3),padding='same', activation='relu',input_shape=(32,32,3)))
    model.add(Conv2D(32,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), padding='same', activation='relu'))
    model.add(Conv2D(64,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax'))
    '''''
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    '''''
    return model
model=create_model()

print(model.summary())
