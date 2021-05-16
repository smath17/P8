import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from simple_model.simple_cnn import load_image


def cnn_setup(fx_count, mp_count, class_count, input_shape=(256, 256, 3)):
    """

    :rtype: keras.Model
    :param class_count: Number of classes
    :param mp_count: Number of max-pooling layers {3, 4, 5}
    :param fx_count: Number of feature extraction layers {1, 2}
    :param input_shape: Tuple determining shape of input images (W, H, 3)
    """

    # TODO: Should have input equal to the number of pictures per game (for our model)
    img_inputs = keras.Input(shape=input_shape)

    # Initialize model by normalizing input
    init_batch_norm = keras.layers.BatchNormalization()(img_inputs)
    current_layer = init_batch_norm

    # Filter is referenced to as 'number of kernels' in the paper
    init_filter = 64
    dropout_rate = 0.2

    # Each MaxPool layer is preceded by a number of feature extractions
    for mp in range(1, mp_count + 1):
        for fx in range(fx_count):
            current_layer = feature_extract_layers(current_layer, dropout_rate, init_filter, mp)
        current_layer = layers.MaxPooling2D(strides=2)(current_layer)

    reduction_filter = 2 ** (mp_count - 2) * init_filter
    reduced_layer = dimension_reduction_layers(current_layer, dropout_rate, reduction_filter)
    output = classification_layers(reduced_layer, class_count)

    model = keras.Model(inputs=img_inputs, outputs=output, name="suatap_model")
    return model


def train_model(model: keras.Model, train_ds, val_ds):
    """

    :param model: Model from suatap_model.cnn_setup()
    :param val_ds: Validation set
    :param train_ds: Training set, images should be of size (256, 256 ,3)
    """

    # Learning rate for icons = 0.001, for screenshots = 0.01
    model.compile(keras.optimizers.Adam(learning_rate=0.01), loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.Accuracy(),
                           tf.keras.metrics.AUC(multi_label=True),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top 2 accuracy"),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top 3 accuracy"),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=4, name="top 4 accuracy"),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top 5 accuracy")])
    model.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=[tensorboard_setup()], verbose=2, workers=8,
              use_multiprocessing=True)

    model.save("logs/fit/suatap/model")


def tensorboard_setup():
    log_dir = "logs/fit/suatap/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback


def feature_extract_layers(prev_layer, dropout_rate, init_filter, current_maxpool_count):
    """

    :param current_maxpool_count: The current iteration of max pools as an integer
    :param init_filter: Size of the initial filter used in Conv2D as an integer
    :param prev_layer: Previous layer executed.
    :param dropout_rate: Float between 0 and 1, fraction of units dropped.
    :return:
    """
    conv_filter = 2 ** (current_maxpool_count - 1) * init_filter
    # Padding=same means the input and output shape of the conv2d layer will be the same
    conv = layers.Conv2D(conv_filter, kernel_size=3, use_bias=False, activation=layers.LeakyReLU(), padding='same')(
        prev_layer)
    batch_norm = layers.BatchNormalization()(conv)
    dropout = layers.Dropout(rate=dropout_rate)(batch_norm)
    return dropout


def dimension_reduction_layers(prev_layer, dropout_rate, filter):
    conv1x1 = layers.Conv2D(filter, kernel_size=1, use_bias=False, activation=layers.LeakyReLU())(prev_layer)
    batch_norm = layers.BatchNormalization()(conv1x1)
    dropout = layers.Dropout(dropout_rate)(batch_norm)
    max_pool = layers.MaxPooling2D()(dropout)
    return max_pool


def classification_layers(prev_layer, num_classes):
    flatten = layers.Flatten()(prev_layer)
    fully_2c = layers.Dense(units=num_classes * 2)(flatten)
    fully_c = layers.Dense(units=num_classes)(fully_2c)
    output = layers.Softmax()(fully_c)
    return output


def inference_mode(img_path):
    with open("../resources/tags.txt") as f:
        labels = [line.rstrip('\n') for line in f]

    # TODO: Can be modified to do multiple at once
    images = load_image(img_path, 256)

    model: keras.Model = keras.models.load_model("logs/fit/suatap/model")

    prediction = model(images, training=False)

    # Retrieve top 3 and reverse to get the highest first
    top3 = np.argpartition(prediction[0], -3)[-3:]
    top3 = top3[::-1]

    # Total distribution
    # print(prediction)

    # Print predicted label together with the probability
    counter = 1
    for value in top3:
        print("Top " + str(counter) + ": " + labels[value] + "\nProbability: " + str(prediction[0][value]))
        counter += 1
