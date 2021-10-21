from OCR_CNN import create_model, splitting_data,augmentation
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from tensorflow import keras
from tensorflow.keras import layers,optimizers
from tensorflow.keras.optimizers import Adam
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator




def training_model():
    # initialize model

    x_train, x_test, x_validation, y_train, y_test, y_validation=augmentation()
    model1 = create_model()
    # set training process parameters
    batchSize = 256
    epochs = 10000
    print(y_train.shape)
    label_as_binary = LabelBinarizer()
    y_train = label_as_binary.fit_transform(y_train)
    print(y_train.shape)
    # Set the training configurations: optimizer, loss function, accuracy metrics
    model1.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model1.fit(x_train, y_train, batch_size=batchSize, epochs=epochs,
                         validation_data=(x_validation, y_validation))
    

    return history, model1


x_train, x_test, x_validation, y_train, y_test, y_validation=augmentation()
history,model1=training_model()
print(model1.evaluate(x_test, y_test))

# Loss Curves
plt.figure(1, figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

# Accuracy Curves
plt.figure(2, figsize=[8, 6])
plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
plt.plot(history.history['val_accuracy'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)
plt.show()
