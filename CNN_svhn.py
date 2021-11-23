import numpy as np
from tensorflow import keras
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import joblib
import cv2
# %matplotlib inline

# Set random state

np.random.seed(20)


def load_data():
    # Load the data

    train_raw = loadmat('C:/Users/shahr/Downloads/svhn/train_32x32.mat')
    test_raw = loadmat('C:/Users/shahr/Downloads/svhn/test_32x32.mat')

    # Load images and labels

    train_images = np.array(train_raw['X'])
    test_images = np.array(test_raw['X'])

    train_labels = train_raw['y']
    test_labels = test_raw['y']

    # Check the shape of the data

    print(train_images.shape)
    print(test_images.shape)
    return train_images, train_labels, test_images, test_labels


def modify_dataset():
    train_images, train_labels, test_images, test_labels = load_data()
    # Fix the axes of the images
    train_images = np.moveaxis(train_images, -1, 0)
    test_images = np.moveaxis(test_images, -1, 0)

    print(train_images.shape)
    print(test_images.shape)

    # Plot a random image and its label

    # plt.imshow(train_images[10529])
    # plt.show()

    # print('Label: ', train_labels[10529])

    # Convert train and test images into 'float64' type

    train_images = train_images.astype('float64')
    test_images = test_images.astype('float64')

    # Convert train and test labels into 'int64' type

    train_labels = train_labels.astype('int64')
    test_labels = test_labels.astype('int64')
    return train_images, train_labels, test_images, test_labels


def normalize():
    train_images, train_labels, test_images, test_labels = modify_dataset()
    # Normalize the images data

    print('Min: {}, Max: {}'.format(train_images.min(), train_images.max()))

    train_images /= 255.0
    test_images /= 255.0
    return train_images, train_labels, test_images, test_labels


def splitting_data():
    train_images, train_labels, test_images, test_labels = normalize()
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)
    # Split train data into train and validation sets

    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels,
                                                      test_size=0.15, random_state=22)

    y_val.shape
    return X_train, X_val, y_train, y_val,test_labels


def data_augmentation():
    # Data augmentation

    datagen = ImageDataGenerator(rotation_range=8,
                                 zoom_range=[0.95, 1.05],
                                 height_shift_range=0.10,
                                 shear_range=0.15)
    return datagen


def create_model():
    # Define actual model

    keras.backend.clear_session()

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same',
                            activation='relu',
                            input_shape=(32, 32, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), padding='same',
                            activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(64, (3, 3), padding='same',
                            activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), padding='same',
                            activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),

        keras.layers.Conv2D(128, (3, 3), padding='same',
                            activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, (3, 3), padding='same',
                            activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.3),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model


# model=create_model()

def prediction():
    # Fit model in order to make predictions
    train_images, train_labels, test_images, test_labels = normalize()

    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)
    print(train_labels.shape,'train')
    print(test_labels.shape, 'test')

    X_train, X_val, y_train, y_val,test_labels = splitting_data()
    datagen = data_augmentation()
    model = create_model()
    print(y_train.shape,'y_train')
    print(test_labels.shape,'test labels')
    early_stopping = keras.callbacks.EarlyStopping(patience=8)
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        '/kaggle/working/best_cnn.h5',
        save_best_only=True)

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                                  epochs=10, validation_data=(X_val, y_val),
                                  callbacks=[early_stopping, model_checkpoint])
    # save the model to disk
    #filename = 'finalized_model.sav'
    #joblib.dump(history, filename)
    # Evaluate train and validation accuracies and losses

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Visualize epochs vs. train and validation accuracies and losses

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Epochs vs. Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Epochs vs. Training and Validation Loss')

    #plt.show()
    plt.savefig('Loss&Accuracy.png')
    # Evaluate model on test data
    test_loss, test_acc = model.evaluate(x=test_images, y=test_labels, verbose=0)

    print('Test accuracy is: {:0.4f} \nTest loss is: {:0.4f}'.
          format(test_acc, test_loss))

    # Get predictions and apply inverse transformation to the labels

    y_pred = model.predict(X_train)
    y_pred = lb.inverse_transform(y_pred, lb.classes_)
    y_train = lb.inverse_transform(y_train, lb.classes_)

    test_pred = model.predict(test_images)
    test_pred = lb.inverse_transform(test_pred, lb.classes_)
    test_labels = lb.inverse_transform(test_labels, lb.classes_)
    # Plot the confusion matrix
    print(y_pred,'y_pred')
    print(y_train,'y_train')
    print(test_pred,'test pred')
    print(test_labels,'test label')
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Epochs vs. Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Epochs vs. Training and Validation Loss')

    #plt.show()
    plt.savefig('Loss&Accuracy.png')
    matrix1 = confusion_matrix(y_train, y_pred, labels=lb.classes_)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix1, annot=True, cmap='Greens', fmt='d', ax=ax)
    plt.title('Confusion Matrix for training dataset')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    #plt.show()
    plt.savefig('CM_train.png')
    #print(test_labels)
    matrix2 = confusion_matrix(test_labels, test_pred, labels=lb.classes_)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix2, annot=True, cmap='Blues', fmt='d', ax=ax)
    plt.title('Confusion Matrix for test dataset')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    #plt.show()
    plt.savefig('CM_test.png')

    return matrix1, matrix2, history


matrix1, matrix2,history = prediction()
