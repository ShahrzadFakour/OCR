import warnings;
from random import random

import cv2
import random
import numpy as np
import seaborn as sns
import tensorflow.python.keras.engine.training
from sklearn.metrics import confusion_matrix
#from tensorflow.python.keras.utils.np_utils import to_categorical
from datasets import splitDatasets, one_hot_encoding, lists, fw
from model import cnn_model
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
    model,y=cnn_model()

    checkpointer=ModelCheckpoint(filepath=fw,save_best_only=True)
    lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=8,factor=.75)
    estopping=EarlyStopping(monitor='val_loss',patience=8)
    history=model.fit(x_train,y_train_list,
                          validation_data=(x_valid,y_valid_list),
                          epochs=20,batch_size=128,verbose=1,
                          callbacks=[checkpointer,lr_reduction,estopping])

    #    pickle.dump(model, file)
    filename = 'finalized_model.h5'
    model.save(filename)
    model.load_weights(fw)
    cnn_scores=model.evaluate(x_test,y_test_list)
    print("CNN. Scores: \n" ,(cnn_scores))
    print("First digit. Accuracy: %.2f%%"%(cnn_scores[6]*100))
    print("Second digit. Accuracy: %.2f%%"%(cnn_scores[7]*100))
    print("Third digit. Accuracy: %.2f%%"%(cnn_scores[8]*100))
    print("Fourth digit. Accuracy: %.2f%%"%(cnn_scores[9]*100))
    print("Fifth digit. Accuracy: %.2f%%"%(cnn_scores[10]*100))
    avg_accuracy=sum([cnn_scores[i] for i in range(6,11)])/5
    print("CNN Model. Average Accuracy: %.2f%%"%(avg_accuracy*100))
    pl.figure(figsize=(11,5));
    k=10
    keys=list(history.history.keys())[17:]
    print(keys)
    pl.plot(history.history[keys[0]][k:],label='First digit')
    pl.plot(history.history[keys[1]][k:],label='Second digit')
    pl.plot(history.history[keys[2]][k:],label='Third digit')
    pl.plot(history.history[keys[3]][k:],label='Fourth digit')
    pl.plot(history.history[keys[4]][k:],label='Fifth digit')
    pl.legend(); pl.title('Accuracy');
    plt.savefig('Accuracy.png')
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

    #test_loss, test_acc = cnn_model.evaluate(x=x_test, y=y_test_list, verbose=0)

    #print('Test accuracy is: {:0.4f} \nTest loss is: {:0.4f}'.
         # format(test_acc, test_loss))



    #y_pred = y
    #print(y_pred)
    #y_pred_cls=tensorflow.math.argmax(y_pred, axis=1)

    #y_pred = np.argmax(to_categorical(y_pred, y_train_list))
    #print(y_pred.shape)
    print(y_train[5268])
    print(np.argmax(y_train[5268],axis=1))
    print(len(x_test),'x_test')

    test_pred=model.predict(x_test)
    print(np.shape(test_pred))
    test_pred=np.transpose(test_pred,[1,0,2])
    #print(test_pred[0],([np.argmax(test_pred[0][i])for i in range(0,len(test_pred[0]))]))
    #print(test_pred[5],(np.argmax(test_pred[5][i]) for i in range(0,5)))
    #print(test_pred[10],(np.argmax(test_pred[10][i]) for i in range(0,5)))
    #test_pred=np.transpose(test_pred,[1,0,2])
    #print(np.shape(test_pred))
    #print(test_pred[0],'[0]')
    #print(test_pred[0][0], '[0][0]')
    #test_pred=[np.argmax(test_pred[0][i]) for i in range(0,len(test_pred[0]))]
    prediction=[]
    for i in range(0,len(test_pred)):
        lable = []
        for j in range(0,5):
            real_label=np.argmax(test_pred[i][j])
            lable.append(real_label)
        prediction.append(lable)
    np.asarray(prediction)
    print(len(prediction))
    #print(prediction,'test prediction')
    #test_pred1 = [np.argmax(test_pred[0][0][i]) for i in range(0,len(test_pred[0]))]
    #print((np.argmax(test_pred[0][i]) for i in range(0,len(test_pred[0]))),'argmax')
    #test_pred=np.array(test_pred).transpose()
    #cv2.imshow('predict', y_pred)
    #cv2.waitKey(0)
    #print(y_pred)
    #print(test_pred)
    #test_pred=[np.argmax(test_pred[0][i]) for i in range(0,len(test_pred[0]))]

    #print(test_pred1)
    print(np.argmax(y_test[1052],axis=1))
    #print(np.argmax(y_test[1052]))
    test_label=[np.argmax(y_test[i],axis=1) for i in range(0,len(y_test))]


    #y_train = np.argmax(to_categorical(y_train, y_train_list))
    #test_pred = cnn_model.predict(y_test)
    #test_pred = np.argmax(to_categorical(test_pred, y_test_list))
    #test_labels = np.argmax(to_categorical(y_test, y_test_list))

    # Find the position of the non missing labels
    #non_zero = np.where(test_label.flatten() != 10)
    img=x_test[1000]
    cv2.imshow('rand img',img)
    cv2.waitKey(0)
    print(prediction[1000],'predicted label')
    print(test_label[1000],'real label')


    #not_none=np.where(np.ndarray.flatten(prediction)=!10)
    '''''
    correct=np.array([a==b].all() for a,b in zip(test_pred,prediction))
    print(correct)
    images = x_test[correct]
    cls_true = y_test[correct]
    cls_pred = test_pred[correct]
    plt.show(images, 6, 6, cls_true, cls_pred)
    
    '''''
    for i in range(10):
        rand=random.randint(0,3200)
        comparison = test_label[rand] == prediction[rand]
        equal_arrays = comparison.all()
        print(equal_arrays)
    lables = [i for i in range(11)]
    test_label=np.array(test_label)
    prediction = np.array(prediction)
    matrix1 = confusion_matrix(test_label.flatten(), prediction.flatten(),labels=lables)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix1, annot=True, cmap='Greens', fmt='d', ax=ax)
    plt.title('Confusion Matrix for test dataset')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    plt.savefig('CM_test.png')
    # print(test_labels)
    '''''
    matrix2 = confusion_matrix(test_labels, test_pred, labels=[i for i in range(0, 11)])

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix2, annot=True, cmap='Blues', fmt='d', ax=ax)
    plt.title('Confusion Matrix for test dataset')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    return history
    '''''

history=train()

