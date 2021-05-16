import datetime
from os import path

import keras.initializers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import visualkeras
from PIL import ImageFont
from keras.models import load_model
from keras.preprocessing.image import img_to_array
# make a prediction for a new image.
from keras.preprocessing.image import load_img
from tensorflow.keras import layers, models


def train_cnn_model(train_set, val_set, name):
    """
    Trains and saves a CNN model. Arguments should be from load_data.

    :param name: Name of the model, for saving purposes
    :param train_set: Dataset used for training
    :param val_set: Dataset used for validation
    """
    # Do not show version warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    num_classes = len(train_set.class_indices.items())
    dropout_conv = 0.2
    dropout_dense = 0.5

    model = models.Sequential(layers=(
        layers.experimental.preprocessing.Rescaling(scale=1. / 127.5, offset=-1),  # Scale to [-1, 1]
        layers.Conv2D(32, 3, activation='relu', padding="same", kernel_initializer=keras.initializers.HeUniform()),
        layers.Dropout(dropout_conv),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding="same", kernel_initializer=keras.initializers.HeUniform()),
        layers.Dropout(dropout_conv),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding="same", kernel_initializer=keras.initializers.HeUniform()),
        layers.Dropout(dropout_conv),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_initializer=keras.initializers.HeUniform()),
        layers.Dropout(dropout_dense),
        layers.Dense(num_classes),
        layers.Activation(keras.activations.softmax))
    )

    # Experimental lr = 1e-6
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.Accuracy(),
                           tf.keras.metrics.AUC(multi_label=True),
                           tf.keras.metrics.Recall()])

    model.fit(train_set, epochs=100, validation_data=val_set, callbacks=[tensorboard_setup(name)], verbose=2, workers=8, use_multiprocessing=True)

    model.save("cnn/model/" + name)


def __generate_model_figure(model):
    """
    Generates a figure of the keras sequential model.
    layers.InputLayer(input_shape=(32, 32, 3)) must be the first layer of the model
    """
    font = ImageFont.truetype("arial.ttf", 20)
    visualkeras.layered_view(model, legend=True, to_file='output3.png', font=font)


def simple_evaluate(test_data):
    model: keras.Model = keras.models.load_model('cnn.model')
    model.evaluate(test_data, workers=4)


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
