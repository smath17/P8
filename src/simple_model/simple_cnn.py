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
    weight_init_seed = 15

    model = models.Sequential(layers=(
        layers.experimental.preprocessing.Rescaling(scale=1. / 127.5, offset=-1),  # Scale to [-1, 1]
        layers.Conv2D(32, 3, activation='relu', padding="same",
                      kernel_initializer=keras.initializers.HeUniform(seed=weight_init_seed)),
        layers.Dropout(0.2),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding="same",
                      kernel_initializer=keras.initializers.HeUniform(seed=weight_init_seed)),
        layers.Dropout(0.2),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding="same",
                      kernel_initializer=keras.initializers.HeUniform(seed=weight_init_seed)),
        layers.Dropout(0.2),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_initializer=keras.initializers.HeUniform(seed=weight_init_seed)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, kernel_initializer=keras.initializers.glorot_uniform(seed=weight_init_seed)),
        layers.Activation(keras.activations.sigmoid))
    )

    # Experimental lr = 1e-6
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(),  # Also known as Top 1
                           tf.keras.metrics.AUC(multi_label=True),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top 2 accuracy"),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top 3 accuracy"),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=4, name="top 4 accuracy"),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top 5 accuracy")])

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    model.fit(train_set, epochs=100, validation_data=val_set, callbacks=[tensorboard_setup(name), stop_early],
              verbose=2, workers=8, use_multiprocessing=True)

    model.save("cnn/model/" + name)


def __generate_model_figure(model):
    """
    Generates a figure of the keras sequential model.
    layers.InputLayer(input_shape=(32, 32, 3)) must be the first layer of the model
    """


def simple_evaluate(test_data):
    model: keras.Model = keras.models.load_model('cnn.model')
    model.evaluate(test_data, workers=4)


def tensorboard_setup(model_name):
    log_dir = "logs/fit/simple/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_" + model_name)
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
