import keras
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.applications import MobileNetV2
from keras import regularizers

def MobileNetV2_model(learning_rate, input_shape,class_number):
    baseModel = MobileNetV2(include_top=False, input_tensor=keras.layers.Input(shape=input_shape))
    for layer in baseModel.layers[:-4]:
        layer.trainable = False
    model = keras.models.Sequential()
    model.add(baseModel)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu',
                                bias_regularizer=regularizers.L2(1e-4)))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))
    return model

lr = 0.0001
size = (224, 224)
shape = (224,224, 3) 
epochs = 20
class_number = 3

model = MobileNetV2_model(lr,shape,class_number)
model.compile(loss= "categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

seed = 1
train = keras.utils.image_dataset_from_directory(
    "download/",
    seed=1,
    image_size=(224,224),
    batch_size=32,
    validation_split = 0.2,
    subset = "training",
    label_mode='categorical'
)
valid = keras.utils.image_dataset_from_directory(
    "download/",
    seed=1,
    image_size=(224,224),
    batch_size=32,
    validation_split = 0.2,
    subset = "validation",
    label_mode='categorical'
)
#test = keras.utils.image_dataset_from_directory(
#    "download/",
#    seed=1,
#    image_size=(224,224),
#    batch_size=32,
#    label_mode='categorical'
#)

history = model.fit(
    train,
    epochs=epochs,
    validation_data=valid,
    batch_size=32
)

#predictions = np.array([])
#labels =  np.array([])
#for x, y in test:
#    predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis = -1)])
#    labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
#
#print(tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy())