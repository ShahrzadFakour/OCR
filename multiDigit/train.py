import warnings;

import numpy as np
import seaborn as sns
import tensorflow.python.keras.engine.training
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils.np_utils import to_categorical

from datasets import splitDatasets, one_hot_encoding, lists, fw
import model
import h5py
warnings.filterwarnings('ignore')

import pylab as pl
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


def train():
    X, y = one_hot_encoding()
    x_train, x_valid, x_test, y_train, y_valid, y_test = splitDatasets(X, y)
    y_train_list, y_test_list, y_valid_list=lists()
    cnn_model=model.cnn_model()

    checkpointer=ModelCheckpoint(filepath=fw,verbose=1,save_best_only=True)
    lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=8,verbose=2,factor=.75)
    estopping=EarlyStopping(monitor='val_loss',patience=8,verbose=1)
    history=cnn_model.fit(x_train,y_train_list,
                          validation_data=(x_valid,y_valid_list),
                          epochs=2,batch_size=128,verbose=1,
                          callbacks=[checkpointer,lr_reduction,estopping])

    #    pickle.dump(model, file)
    filename = 'finalized_model.h5'
    cnn_model.save(filename)
    cnn_model.load_weights(fw)
    cnn_scores=cnn_model.evaluate(x_test,y_test_list,verbose=0)
    print("CNN. Scores: \n" ,(cnn_scores))
    print("First digit. Accuracy: %.2f%%"%(cnn_scores[6]*100))
    print("Second digit. Accuracy: %.2f%%"%(cnn_scores[7]*100))
    print("Third digit. Accuracy: %.2f%%"%(cnn_scores[8]*100))
    print("Fourth digit. Accuracy: %.2f%%"%(cnn_scores[9]*100))
    print("Fifth digit. Accuracy: %.2f%%"%(cnn_scores[10]*100))
    avg_accuracy=sum([cnn_scores[i] for i in range(6,11)])/5
    print("CNN Model. Average Accuracy: %.2f%%"%(avg_accuracy*100))
    pl.figure(figsize=(11,5)); k=10
    keys=list(history.history.keys())[17:]
    pl.plot(history.history[keys[0]][k:],label='First digit')
    pl.plot(history.history[keys[1]][k:],label='Second digit')
    pl.plot(history.history[keys[2]][k:],label='Third digit')
    pl.plot(history.history[keys[3]][k:],label='Fourth digit')
    pl.plot(history.history[keys[4]][k:],label='Fifth digit')
    pl.legend(); pl.title('Accuracy');

    # Evaluate train and validation accuracies and losses

    # train_acc = history.history['accuracy']
    # val_acc = model.history['val_accuracy']
    #
    # train_loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # # Visualize epochs vs. train and validation accuracies and losses
    #
    # plt.figure(figsize=(20, 10))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(train_acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.legend()
    # plt.title('Epochs vs. Training and Validation Accuracy')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(train_loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.legend()
    # plt.title('Epochs vs. Training and Validation Loss')
    #
    # plt.show()
    # plt.savefig('Loss&Accuracy.png')
    # Evaluate model on test data
    test_loss, test_acc = cnn_model.evaluate(x=x_test, y=y_test_list, verbose=0)

    print('Test accuracy is: {:0.4f} \nTest loss is: {:0.4f}'.
          format(test_acc, test_loss))
    plt.figure(figsize=(11, 5));
    k = 10
    keys = list(history.history.keys())[17:]
    plt.plot(history.history[keys[0]][k:], label='First digit')
    plt.plot(history.history[keys[1]][k:], label='Second digit')
    plt.plot(history.history[keys[2]][k:], label='Third digit')
    plt.plot(history.history[keys[3]][k:], label='Fourth digit')
    plt.plot(history.history[keys[4]][k:], label='Fifth digit')
    plt.legend();
    plt.title('Accuracy');
    plt.savefig('Accuracy.png')

    y_pred = model.predict(x_train)
    y_pred = np.argmax(to_categorical(y_pred, y_train_list))
    y_train = np.argmax(to_categorical(y_train, y_train_list))
    test_pred = model.predict(y_test)
    test_pred = np.argmax(to_categorical(test_pred, y_test_list))
    test_labels = np.argmax(to_categorical(y_test, y_test_list))

    matrix1 = confusion_matrix(y_train, y_pred, labels=[i for i in range(0, 11)])

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix1, annot=True, cmap='Greens', fmt='d', ax=ax)
    plt.title('Confusion Matrix for training dataset')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # plt.show()
    # plt.savefig('CM_train.png')
    # print(test_labels)
    matrix2 = confusion_matrix(test_labels, test_pred, labels=[i for i in range(0, 11)])

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix2, annot=True, cmap='Blues', fmt='d', ax=ax)
    plt.title('Confusion Matrix for test dataset')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    return history
history=train()

