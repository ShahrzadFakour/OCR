import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical



def read_digits():
    # Path to the data directory
    path=r'C:/Users/shahr/Downloads/archive/Exp_10'

    count = 0
    images = []  # LIST CONTAINING ALL THE IMAGES
    classNo = []  # LIST CONTAINING ALL THE CORRESPONDING CLASS ID OF IMAGES
    myList = os.listdir(path)
    print("Total Classes Detected:", len(myList))
    noOfClasses = len(myList)
   # print(noOfClasses)
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
    print(x_train.shape)

    print('YYYYYYYYYtrain',y_train.shape)

    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=testRatio)
    print(x_train.shape)
    print(x_test.shape)
    print(x_validation.shape)
    print('YYYYYYYYYtrain',y_train.shape)
    return x_train, x_test, x_validation, y_train, y_test, y_validation

def pre_processing(digit):
    digit = cv2.cvtColor(digit, cv2.COLOR_RGB2GRAY)
    digit = cv2.bitwise_not(digit)
    digit = cv2.equalizeHist(digit)
    digit = digit / 255
    return digit


def modify_data_sets():
    x_train, x_test, x_validation, y_train, y_test, y_validation = splitting_data()
    # print(len(x_train))
    # date = pre_processing(x_train)
    x_train = np.array(list(map(pre_processing, x_train)))
    x_test = np.array(list(map(pre_processing, x_test)))
    x_validation = np.array(list(map(pre_processing, x_validation)))
    print(x_train.shape)
    print(x_test.shape)
    print(x_validation.shape)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2],
                                        1)  # add depth for each image
    print(x_train.shape)
    print(x_test.shape)
    print(x_validation.shape)
    # print(type(x_train))
    return x_train, x_test, x_validation, y_train, y_test, y_validation


def augmentation():
    # augmentation
    x_train, x_test, x_validation, y_train, y_test, y_validation= modify_data_sets()

    data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                                        rotation_range=10)
    data_generator.fit(x_train)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    y_validation = to_categorical(y_validation, 10)
    print(type(y_train))
    return x_train, x_test, x_validation, y_train, y_test, y_validation
    
def create_model():
    model=Sequential()
    model.add(Conv2D(32,(3,3),padding='same', activation='relu',input_shape=(32,32,1)))
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
    model.add(Dense(10))
    return model
model=create_model()

print(model.summary())
