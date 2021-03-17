from tensorflow import keras
from tensorflow.keras import layers


# TODO: Figure out parameters from the paper
def cnn(num_extractions, num_pools, num_classes, input_shape=(256, 256, 3)):
    """

    :param num_classes: Number of classes
    :param num_pools: Number of max-pooling layers
    :param num_extractions: Number of feature extractions to be done for each pooling layer
    :param input_shape: Tuple determining shape of input images (W, H, 3)
    """

    # TODO: Should have input equal to the number of pictures per game (for our model)
    img_inputs = keras.Input(shape=input_shape)

    # Initialize model by normalizing input
    init_batch_norm = keras.layers.BatchNormalization()(img_inputs)
    save_layer = init_batch_norm

    # TODO: figure out the number of kernels (filter)
    kernels = input_shape[0]
    for m in range(num_pools):
        for n in range(1, num_extractions + 1):
            save_layer = feature_extract_layers(save_layer, 0.5, kernels * (2 ** m))
        layers.MaxPooling2D(strides=2)(save_layer)

    reduced_layer = dimension_reduction_layers(save_layer, 0.5)
    output = classification_layers(reduced_layer, num_classes)

    model = keras.Model(inputs=img_inputs, outputs=output, name="suatap_model")
    model.summary()

    # keras.utils.plot_model(model, "suatap_model.png")


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


def dimension_reduction_layers(prev_layer, dropout_rate):
    conv1x1 = layers.Conv2D(1, kernel_size=1, use_bias=False, activation=layers.LeakyReLU())(prev_layer)
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


cnn(2, 1, 10)
