import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime


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

    init_filter = 64
    dropout_rate = 0.2

    # Each MaxPool layer is preceded by a number of feature extractions
    for mp in range(mp_count):
        for fx in range(fx_count):
            current_layer = feature_extract_layers(current_layer, dropout_rate, init_filter * (2 ** mp))
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
    train_ds, val_ds = prepare_data(train_ds, val_ds)

    # Learning rate for icons = 0.001, for screenshots = 0.01
    model.compile(keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[tensorboard_setup()], verbose=2)


def tensorboard_setup():
    log_dir = "logs/fit/suatap" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback


def prepare_data(train_ds, val_ds):
    # Cache data to avoid I/O bottleneck
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


def feature_extract_layers(prev_layer, dropout_rate, filter):
    """

    :param filter: Size of the filter used in Conv2D
    :param prev_layer: Previous layer executed.
    :param dropout_rate: Float between 0 and 1, fraction of units dropped.
    :return:
    """
    conv = layers.Conv2D(filter, kernel_size=3, use_bias=False, activation=layers.LeakyReLU())(prev_layer)
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
