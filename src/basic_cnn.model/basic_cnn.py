from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
"""
img = image.load_img("C:/Users/MadPe/Documents/AAU/Kandidat/8 semester/P8/src/basic_cnn.model/train/Racing/4290.jpg")
plt.imshow(img)
plt.show()

print(cv2.imread("C:/Users/MadPe/Documents/AAU/Kandidat/8 semester/P8/src/basic_cnn.model/train/Racing/4290.jpg").shape)

train = ImageDataGenerator(rescale= 1/255)
validation =ImageDataGenerator(rescale= 1/255)

train_dataset = train.flow_from_directory('C:/Users/MadPe/Documents/AAU/Kandidat/8 semester/P8/src/basic_cnn.model/train/',
                                          target_size= (200,200),
                                          batch_size= 25,
                                          class_mode= 'binary')

validation_dataset = train.flow_from_directory('C:/Users/MadPe/Documents/AAU/Kandidat/8 semester/P8/src/basic_cnn.model/validation/',
                                          target_size= (200,200),
                                          batch_size= 25,
                                          class_mode= 'binary')

print(train_dataset.class_indices)

print(train_dataset.classes)

model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation= 'relu', input_shape= (200,200,3)),
                                     tf.keras.layers.MaxPooling2D(2,2),
                                     #
                                     tf.keras.layers.Conv2D(32,(3,3),activation= 'relu'),
                                     tf.keras.layers.MaxPooling2D(2,2),
                                     #
                                     tf.keras.layers.Conv2D(64,(3,3),activation= 'relu'),
                                     tf.keras.layers.MaxPooling2D(2,2),
                                     ##
                                     tf.keras.layers.Flatten(),
                                     ##
                                     tf.keras.layers.Dense(512,activation= 'relu'),
                                     ##
                                     tf.keras.layers.Dense(1,activation= 'sigmoid')
                                     ])
#model.compile(loss= 'binary_crossentropy', optimizer= RMSprop(lr=0.001),
model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

model.fit(train_dataset,
          steps_per_epoch= 80,
          epochs= 10,
          validation_data= validation_dataset)

model.save("basic.cnn")
"""

dir_path = 'C:/Users/MadPe/Documents/AAU/Kandidat/8 semester/P8/src/basic_cnn.model/test/'
model = load_model('basic.cnn')

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+i, target_size=(200,200))
    plt.imshow(img)

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis =0)
    images = np.vstack([X])
    val = model.predict(images)
    if val == 0:
        plt.title("predicted Racing")
    else:
        plt.title("predicted Strategy")
    plt.show()
