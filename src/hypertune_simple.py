import kerastuner as kt
from tensorflow import keras
from tensorflow.keras import layers


def build_model(hp: kt.HyperParameters):
    weight_init_seed = 15
    num_classes = 23

    # Find optimal value
    # TODO: hp_dense = hp.Int('dense_units', min_value=32, max_value=128, step=32)
    # hp_dense = 64
    # TODO: hp_drop_conv = hp.Choice('conv_drop', values=[0.2, 0.3, 0.5, 0.8])
    hp_drop_conv = 0.2
    # TODO: hp_drop_dense = hp.Choice('dense_drop', values=[0.5, 0.8])
    hp_drop_dense = 0.5
    # Choose optimal value from selection
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

    model = keras.models.Sequential(layers=(
        layers.experimental.preprocessing.Rescaling(scale=1. / 127.5, offset=-1),  # Scale to [-1, 1]
        layers.Conv2D(32, 3, activation='relu', padding="same",
                      kernel_initializer=keras.initializers.HeUniform(seed=weight_init_seed)),
        layers.Dropout(hp_drop_conv),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding="same",
                      kernel_initializer=keras.initializers.HeUniform(seed=weight_init_seed)),
        layers.Dropout(hp_drop_conv),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding="same",
                      kernel_initializer=keras.initializers.HeUniform(seed=weight_init_seed)),
        layers.Dropout(hp_drop_conv),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu',
                     kernel_initializer=keras.initializers.HeUniform(seed=weight_init_seed)),
        layers.Dropout(hp_drop_dense),
        layers.Dense(num_classes, kernel_initializer=keras.initializers.glorot_uniform(seed=weight_init_seed)),
        layers.Activation(keras.activations.sigmoid))
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.CategoricalAccuracy(),
                           keras.metrics.AUC(multi_label=True),
                           keras.metrics.Recall(),
                           keras.metrics.Precision()])

    return model


def tune_params(train_ds, val_ds):

    tuner = kt.Hyperband(build_model,
                         objective='val_loss',
                         max_epochs=20,
                         factor=3,
                         directory='my_dir',
                         project_name='tune_lr')

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(train_ds, validation_data=val_ds, epochs=100, callbacks=[stop_early, tensorboard_setup()], verbose=2)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Learning rate best is:" + best_hps.get('learning_rate'))
    # print("Best amount of units in the second last dense layer is:" + best_hps.get('dense_units'))


def tensorboard_setup():
    log_dir = "logs/fit/hypertune/" + "learning_rate"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback
