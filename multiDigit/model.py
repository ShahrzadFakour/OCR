from tensorflow.keras import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization


def cnn_model():
    model_input=Input(shape=(32,32,1))
    x=BatchNormalization()(model_input)

    x=Conv2D(32,(3,3),activation='relu',padding='same')(model_input)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x=Conv2D(32,(3,3),activation='relu')(x)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x=Dropout(.25)(x)
    x=Conv2D(64,(3,3),activation='relu')(x)
    x=Conv2D(64,(3,3),activation='relu')(x)
    x=Dropout(.25)(x)
    x=Conv2D(196,(3,3),activation='relu')(x)
    x=Dropout(.25)(x)
    x=Flatten()(x)
    x=Dense(512,activation='relu')(x)
    x=Dropout(.5)(x)
    y=[Dense(11,activation='softmax')(x) for i in range(5)]
    model=Model(inputs=model_input,outputs=y)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model
