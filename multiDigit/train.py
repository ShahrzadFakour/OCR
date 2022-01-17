import warnings;

from datasets import tts, one_hot_encoding, lists, fw
import model

warnings.filterwarnings('ignore')

import pylab as pl
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


def train():
    X, y = one_hot_encoding()
    x_train, x_valid, x_test, y_train, y_valid, y_test = tts(X, y)
    y_train_list, y_test_list, y_valid_list=lists()
    cnn_model=model.cnn_model()
    checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
    lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=10,
                                   verbose=2,factor=.75)
    estopping=EarlyStopping(monitor='val_loss',patience=16,verbose=1)
    history=cnn_model.fit(x_train,y_train_list,
                          validation_data=(x_valid,y_valid_list),
                          epochs=100,batch_size=128,verbose=1,
                          callbacks=[checkpointer,lr_reduction,estopping])

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

    return history
history=train()


