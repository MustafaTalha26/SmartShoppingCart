import cv2
import keras
import time
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
import mobilenetv2
from keras.applications import MobileNetV2

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

size = (224, 224)
shape = (224,224, 3) 
epochs = 10
class_number = 3
model = mobilenetv2.MobileNetV2_model(shape,class_number)
model.load_weights('my_model_weights.h5')

while True:
    ret, img= cap.read()
    copyimg = img.copy()
    img = cv2.resize(img, size)
    img = np.expand_dims(img, axis=0)
    y_pred = model.predict(img)

    predicted_class_index = np.argmax(y_pred, axis=1)[0]
    class_labels = ["blackpencil", "singularbanana", "white pug pup"] 
    predicted_class = class_labels[predicted_class_index]
    print(f"Predicted Class: {predicted_class}")
    cv2.putText(copyimg, predicted_class,(10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
    cv2.imshow('Webcam', copyimg)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()