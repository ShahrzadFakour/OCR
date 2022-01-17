import glob

import numpy as np
import pandas as pd
import pylab as pl
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical

path = 'C:\\Users\\shahr\\PycharmProjects\\pythonProject\\tesi\\input\\'
fw = 'weights.housenumbers.hdf5'
glob.glob(path + '*')


def load_datasets():
    train_images = pd.read_csv(path + 'train_images.csv')
    # print(train_images[:10])
    train_labels = pd.read_csv(path + 'train_labels.csv')
    test_images = pd.read_csv(path + 'test_images.csv')
    test_labels = pd.read_csv(path + 'test_labels.csv')
    extra_images = pd.read_csv(path + 'extra_images.csv')
    extra_labels = pd.read_csv(path + 'extra_labels.csv')
    # print('Train images:', train_images.shape)
    # print('Train lables:', train_labels.shape)
    # print('Test images:', test_images.shape)
    # print('Test labels:', test_labels.shape)
    # print('Extra images:', extra_images.shape)
    # print('Etra lables', extra_labels.shape)
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return train_images, train_labels, test_images, test_labels, extra_images, extra_labels


def modify_datasets():
    train_images, train_labels, test_images, test_labels, extra_images, extra_labels = load_datasets()
    train_images = train_images.iloc[:, 1:]
    train_images = train_images.to_numpy().astype('float32').reshape(-1, 32, 32, 1)
    train_labels = train_labels.iloc[:, 1:]
    train_labels = train_labels.to_numpy().astype('int16')
    test_images = test_images.iloc[:, 1:]
    test_images = test_images.to_numpy().astype('float32').reshape(-1, 32, 32, 1)
    test_labels = test_labels.iloc[:, 1:]
    test_labels = test_labels.to_numpy().astype('int16')
    extra_images = extra_images.iloc[:, 1:]
    extra_images = extra_images.to_numpy().astype('float32').reshape(-1, 32, 32, 1)
    extra_labels = extra_labels.iloc[:, 1:]
    extra_labels = extra_labels.to_numpy().astype('int16')
    # print('Train images:', train_images.shape)
    # print('Train lables:', train_labels.shape)
    # print('Test images:', test_images.shape)
    # print('Test labels:', test_labels.shape)
    # print('Extra images:', extra_images.shape)
    # print('Etra lables', extra_labels.shape)
    return train_images, train_labels, test_images, test_labels, extra_images, extra_labels

def one_hot_encoding():
    train_images, train_labels, test_images, test_labels, extra_images, extra_labels=modify_datasets()
    ctrain_labels = to_categorical(train_labels, num_classes=11).astype('int16')
    ctest_labels = to_categorical(test_labels, num_classes=11).astype('int16')
    cextra_labels = to_categorical(extra_labels, num_classes=11).astype('int16')

    n = np.random.randint(1, 2000, 1)[0]
    # print('Label: ', train_labels[n])
    # print(ctrain_labels[n])
    pl.imshow(train_images[n].reshape(32, 32), cmap=pl.cm.bone)

    X = np.concatenate((train_images,
                        test_images), axis=0)
    X = np.concatenate((X, extra_images), axis=0)
    y = np.concatenate((ctrain_labels,
                        ctest_labels), axis=0)
    y = np.concatenate((y, cextra_labels), axis=0)
    return X,y


def tts(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
    n = int(len(x_test) / 2)
    x_valid, y_valid = x_test[:n], y_test[:n]
    x_test, y_test = x_test[n:], y_test[n:]
    print(len(x_train),len(x_test),len(x_valid))
    return x_train, x_valid, x_test, y_train, y_valid, y_test

def lists():
    X,y=one_hot_encoding()
    x_train, x_valid, x_test, y_train, y_valid, y_test = tts(X,y)
    y_train_list = [y_train[:, i] for i in range(5)]
    y_test_list = [y_test[:, i] for i in range(5)]
    y_valid_list = [y_valid[:, i] for i in range(5)]
    for el in [x_train, x_valid, x_test,
               y_train, y_valid, y_test]:
        print(el.shape)
    return y_train_list,y_test_list,y_valid_list
