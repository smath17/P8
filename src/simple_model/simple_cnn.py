from os import path
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
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

    # train_set = train_set.cache().prefetch(buffer_size=AUTOTUNE)
    # test_set = test_set.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(train_set.class_indices.items())

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
        layers.Dense(num_classes, activation="sigmoid"))
    )

    # https://www.tensorflow.org/api_docs/python/tf/keras/metrics
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.Accuracy(),
                           tf.keras.metrics.AUC(multi_label=True),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.CategoricalAccuracy(),
                           tf.keras.metrics.CategoricalCrossentropy()])

    history = model.fit(train_set, epochs=10,
                        validation_data=test_set)

    plt.subplot(3, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend(loc='lower right')

    plt.subplot(3, 2, 2)
    plt.plot(history.history['auc'], label='AUC')
    plt.legend(loc='lower right')

    plt.subplot(3, 2, 3)
    plt.plot(history.history['recall'], label='Recall')
    plt.legend(loc='lower right')

    plt.subplot(3, 2, 4)
    plt.plot(history.history['categorical_accuracy'], label='Categorical Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(3, 2, 5)
    plt.plot(history.history['categorical_crossentropy'], label='Categorical Crossentropy')
    plt.legend(loc='lower right')

    plt.show()

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
