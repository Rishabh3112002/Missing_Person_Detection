import numpy as np
import os
import tensorflow as tf
import PIL
import cv2 as cv
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("[INFO] loading images...")

data = []
labels = []
dir = r'C:\Users\ksr20\PycharmProjects\MissingPersonDetection\Dataset'
cat = ['Criminal', 'Not_Criminal']

for c in cat:
    path = os.path.join(dir, c)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

        data.append(image)
        labels.append(c)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype='float32')
label = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.1, stratify=labels)


print(trainX.shape)
print(testX.shape)

baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = MaxPooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(256, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation='sigmoid')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print('Model Compiling...')
opt = Adam(lr=1e-4, decay=1e-4 / 10)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()
H = model.fit(trainX, trainY, batch_size=16, epochs=20)
model.evaluate(testX, testY)
pred = model.predict(testX, batch_size=1)

pred = np.argmax(pred, axis=1)
print(classification_report(testY.argmax(axis=1), pred, target_names=lb.classes_))

print("[INFO] saving mask detector model...")
os.chdir(r'C:\Users\ksr20\PycharmProjects\MissingPersonDetection')
model.save("criminal_detector.model", save_format="h5")
