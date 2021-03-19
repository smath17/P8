from os import path
import time
import matplotlib.pyplot as plt
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

    # Cache data to avoid I/O bottleneck
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_set = train_set.cache().prefetch(buffer_size=AUTOTUNE)
    test_set = test_set.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = 28

    model = models.Sequential(layers=(
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes))
    )

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_set, epochs=10,
                        validation_data=test_set)

    """
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
    """

    model.save("cnn.model")


# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(32, 32))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 32, 32, 3)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


# load an image and predict the class
def run_example():
    # load the image
    if path.exists("sample_image.jpg"):
        img = load_image('sample_image.jpg')
    else:
        img = load_image('sample_image.png')

    # load model
    model = load_model('cnn.model')
    # predict the class
    result = model.predict_classes(img)
    print(result[0])

