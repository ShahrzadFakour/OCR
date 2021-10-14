import cv2
import numpy as np
from sklearn.model_selection import train_test_split
# from date_generator import create_date
import os
import glob
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Dense, Flatten
from keras.optimizer_v2.adam import Adam


def read_dates():
    img_path = r'C:\Users\shahr\Downloads\archive\Dates'
    dates = []
    names = []
    date = glob.glob(img_path + '/' + '/*.png', recursive=True)
    for d in date:
        dates.append(cv2.imread(d))
        name = os.path.basename(d)
        names.append(name)
        # dates.append(date)

    return dates, names


def spliting_data():
    dates, names = read_dates()
    dates = np.array(dates)
    name = []
    for n in names:
        n = n[:8]
        name.append(n)
    name = np.array(name)
    test_ratio = 0.2
    x_train, x_test, y_train, y_test = train_test_split(dates, name, test_size=test_ratio)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=test_ratio)
    # print(type(dates))
    return x_train, x_test, x_validation, y_train, y_test, y_validation, name, dates


# x_train, x_test, x_validation, y_train, y_test, y_validation = spliting_data()
# print(x_train.shape, x_test.shape, x_validation.shape, y_train.shape, y_test.shape, y_validation.shape)


def pre_processing(date):
    date = cv2.cvtColor(date, cv2.COLOR_RGB2GRAY)
    date = cv2.bitwise_not(date)
    date = cv2.equalizeHist(date)
    date = date / 255
    return date


def modify_data_sets():
    x_train, x_test, x_validation, y_train, y_test, y_validation, name, dates = spliting_data()
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
    return x_train, x_test, x_validation


def augmentation():
    # augmentation
    x_train, x_test, x_validation, y_train, y_test, y_validation, name, date = spliting_data()
    x_train, x_test, x_validation = modify_data_sets()
    data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                                        rotation_range=10)
    data_generator.fit(x_train)
    y_train = to_categorical(y_train, name)
    y_test = to_categorical(y_test, name)
    y_validation = to_categorical(y_validation, name)
    print(type(y_train))
    return y_train, y_test, y_validation


def my_model():
    x_train, x_test, x_validation, y_train, y_test, y_validation, name, dates = spliting_data()
    date_generator = augmentation()
    num_filters = 60
    sizeof_filter1 = (5, 5)
    sizeof_filter2 = (3, 3)
    sizeof_pool = (2, 2)
    num_node = 500
    batchsize_val = 50
    epochs_val = 10
    steps_per_epoch = 2000

    model = Sequential()
    model.add(Conv2D(num_filters, sizeof_filter1, input_shape=(72, 360, 1), activation='relu'))
    model.add(Conv2D(num_filters, sizeof_filter1, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeof_pool))
    model.add(Conv2D(num_filters // 2, sizeof_filter2, activation='relu'))
    model.add(Conv2D(num_filters // 2, sizeof_filter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeof_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(num_node, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_node, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit_generator(date_generator.flow(x_train, y_train, batch_size=batchsize_val),
                                  steps_per_epoch=stepsXepoch,
                                  epochs=epochs_val, validation_data=(x_validation, y_validation),
                                  shuffle=1)

    return model, history


model, history = my_model()
plt.figure(1)
plt.plot(history.history['loss'])

# x_train = modify_trainset()
# print(x_train[30].shape)
# print(type(x_train))
# print(type(x_train))

