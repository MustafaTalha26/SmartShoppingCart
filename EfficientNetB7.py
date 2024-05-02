import keras
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.applications import EfficientNetB7

def EfficientNetB7_model(input_shape,class_number):
    baseModel = EfficientNetB7(include_top=False, input_tensor=keras.layers.Input(shape=input_shape))
    for layer in baseModel.layers[:-4]:
        layer.trainable = False
    model = keras.models.Sequential()
    model.add(baseModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(class_number, activation='softmax'))
    model.compile(loss= "categorical_crossentropy", metrics=["accuracy"], optimizer=keras.optimizers.Adam(1e-4))
    return model

def rest():
    size = (224, 224)
    shape = (224,224, 3) 
    epochs = 10
    class_number = 30

    model = EfficientNetB7_model(shape,class_number)

    seed = 1
    train = keras.utils.image_dataset_from_directory(
        "dataset/",
        seed=1,
        image_size=size,
        batch_size=32,
        validation_split = 0.2,
        subset = "training",
        label_mode='categorical'
    )
    valid = keras.utils.image_dataset_from_directory(
        "dataset/",
        seed=1,
        image_size=size,
        batch_size=32,
        validation_split = 0.2,
        subset = "validation",
        label_mode='categorical'
    )
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    history = model.fit(
        train,
        epochs=epochs,
        validation_data=valid,
        batch_size=32,
        callbacks=[callback]
    )

    model.save_weights('my_model_weights.h5')

def checkdataset():
    size = (224, 224)
    train = keras.utils.image_dataset_from_directory(
        "dataset/",
        seed=1,
        image_size=size,
        batch_size=32,
        validation_split = 0.2,
        subset = "training",
        label_mode='categorical',
        labels='inferred'
    )
    tclasses = train.class_names
    return tclasses

rest()