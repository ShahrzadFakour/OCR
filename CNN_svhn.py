import numpy as np
from tensorflow import keras
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import joblib

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
    return train_images, train_labels, test_images, test_labels


def modify_data():
    train_images, train_labels, test_images, test_labels = load_data()
    # Check the shape of the data
    # print(train_images.shape)
    # print(test_images.shape)

    # Fix the axes of the images
    train_images = np.moveaxis(train_images, -1, 0)
    test_images = np.moveaxis(test_images, -1, 0)

    # print(train_images.shape)
    # print(test_images.shape)
    # Convert train and test images into 'float64' type

    train_images = train_images.astype('float64')
    test_images = test_images.astype('float64')

    # Convert train and test labels into 'int64' type

    train_labels = train_labels.astype('int64')
    test_labels = test_labels.astype('int64')
    return train_images, train_labels, test_images, test_labels


'''''
# Plot a random image and its label

plt.imshow(train_images[10529])
plt.show()

print('Label: ', train_labels[10529])
'''


def normalize():
    train_images, train_labels, test_images, test_labels = modify_data()
    # Normalize the images data

    print('Min: {}, Max: {}'.format(train_images.min(), train_images.max()))

    train_images /= 255.0
    test_images /= 255.0

    # One-hot encoding of train and test labels
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)
    return train_images, train_labels, test_images, test_labels


def splitting_data():
    train_images, train_labels, test_images, test_labels = normalize()

    # Split train data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels,
                                                      test_size=0.15, random_state=22)
    y_val.shape

    return X_train, X_val, y_train, y_val, train_images, train_labels, test_images, test_labels


def data_augmentation():
    # Data augmentation

    datagen = ImageDataGenerator(rotation_range=8,
                                 zoom_range=[0.95, 1.05],
                                 height_shift_range=0.10,
                                 shear_range=0.15)
    return datagen


# Define actual model
def create_model():
    keras.backend.clear_session()
    datagen = data_augmentation()
    X_train, X_val, y_train, y_val, train_images, train_labels, test_images, test_labels = splitting_data()
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

    early_stopping = keras.callbacks.EarlyStopping(patience=8)
    optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        '/kaggle/working/best_cnn.h5',
        save_best_only=True)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Fit model in order to make predictions

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                                  epochs=1, validation_data=(X_val, y_val),
                                  callbacks=[early_stopping, model_checkpoint])
    return model, history


def save_model():
    model, history = create_model()
    filename = 'CNN_model.sav'
    save=joblib.dump(model, filename)
    return save


def evaluate_model():
    X_train, X_val, y_train, y_val, train_images, train_labels, test_images, test_labels = splitting_data()
    model, history = create_model()
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

    # plt.show()

    # Evaluate model on test data
    test_loss, test_acc = model.evaluate(x=test_images, y=test_labels, verbose=0)

    print('Test accuracy is: {:0.4f} \nTest loss is: {:0.4f}'.
          format(test_acc, test_loss))
    # Get predictions and apply inverse transformation to the labels
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)
    y_pred = model.predict(X_train)

    # print(lb.classes_)
    y_pred = lb.inverse_transform(y_pred, lb.classes_)
    y_train = lb.inverse_transform(y_train, lb.classes_)
    print(y_pred)
    print(y_train)
    return y_pred, y_train


def create_confusion_matrix():
    _, _, _, _, _, train_labels, _, test_labels = splitting_data()

    # Plot the confusion matrix
    y_pred, y_train = evaluate_model()
    # lb = LabelBinarizer()
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)
    print(lb.classes_)
    # matrix = confusion_matrix(y_train, y_pred, labels=lb.classes_)
    print(y_pred)
    print(y_train)
    matrix = confusion_matrix(y_train.argmax(axis=1), y_pred.argmax(axis=1), labels=lb.classes_)
    print(matrix)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix, annot=True, cmap='Greens', fmt='d', ax=ax)
    plt.title('Confusion Matrix for training dataset')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()
    return plt


plt = create_confusion_matrix()
'''''
# Get convolutional layers

layers = [model.get_layer('conv2d_1'),
          model.get_layer('conv2d_2'),
          model.get_layer('conv2d_3'),
          model.get_layer('conv2d_4'),
          model.get_layer('conv2d_5'),
          model.get_layer('conv2d_6')]

# Define a model which gives the outputs of the layers

layer_outputs = [layer.output for layer in layers]
activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
'''''
