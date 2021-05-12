import datetime
from os import path

import keras.initializers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
# make a prediction for a new image.
from keras.preprocessing.image import load_img
from tensorflow.keras import layers, models


def train_cnn_model(train_set, test_set):
    """
    Trains and saves a CNN model. Arguments should be from load_data.

    :param train_set: Dataset used for training
    :param test_set: Dataset used for validation
    """
    # Do not show version warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    num_classes = len(train_set.class_indices.items())
    dropout_rate = 0.5

    model = models.Sequential(layers=(
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(32, 3, activation='relu', padding="same", kernel_initializer=keras.initializers.HeUniform),
        layers.Dropout(dropout_rate),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding="same", kernel_initializer=keras.initializers.HeUniform),
        layers.Dropout(dropout_rate),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding="same", kernel_initializer=keras.initializers.HeUniform),
        layers.Dropout(dropout_rate),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_initializer=keras.initializers.HeUniform),
        layers.Dense(num_classes, activation="softmax"))
    )
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.Accuracy(),
                           tf.keras.metrics.AUC(multi_label=True),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top 2 accuracy"),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top 3 accuracy"),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=4, name="top 4 accuracy"),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top 5 accuracy")])

    model.fit(train_set, epochs=100,
              validation_data=test_set, callbacks=[tensorboard_setup()], verbose=2, workers=8)

    model.save("cnn.model")


def tensorboard_setup():
    log_dir = "logs/fit/simple/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback


# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
def load_image(filename, size=32):
    # load the image
    img = load_img(filename, target_size=(size, size))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, size, size, 3)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


def predict_sample_image(class_name_list):
    # load the image
    if path.exists("simple_model/sample_image.jpg"):
        img = load_image('simple_model/sample_image.jpg')
    else:
        img = load_image('simple_model/sample_image.png')

    # load model
    model = load_model('cnn.model')

    # predict the class
    result = model.predict(img)

    # Retrieve top 3 and reverse to get the highest first
    top3 = np.argpartition(result[0], -3)[-3:]
    top3 = top3[::-1]

    # Print predicted label together with the probability
    counter = 1
    for value in top3:
        print("Top " + str(counter) + ": " + class_name_list[value] + "\nProbability: " + str(result[0][value]))
        counter += 1

    # Plot image
    plt.imshow(img[0])
    plt.title(class_name_list[top3[0]])
    plt.show()

    # Print top 1
    print(class_name_list[top3[0]])
