import kerastuner as kt
from tensorflow import keras
from tensorflow.keras import layers


def build_model(hp: kt.HyperParameters):
    # Clear gpu
    keras.backend.clear_session()

    weight_init_seed = 15
    num_classes = 23

    # Find optimal value
    hp_dense_units = hp.Choice('dense_units', values=[num_classes * 2, 64, 128])
    hp_conv_filters_start = hp.Int('conv_filters_start', min_value=32, max_value=128, step=32)
    hp_drop_conv = hp.Choice('conv_drop', values=[0.1, 0.2, 0.5])
    hp_drop_dense = hp.Choice('dense_drop', values=[0.5, 0.8])

    # Choose optimal value from selection
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

    model = keras.models.Sequential(layers=(
        layers.experimental.preprocessing.Rescaling(scale=1. / 127.5, offset=-1),  # Scale to [-1, 1]
        layers.Conv2D(hp_conv_filters_start, 3, activation='relu', padding="same",
                      kernel_initializer=keras.initializers.HeUniform(seed=weight_init_seed)),
        layers.Dropout(hp_drop_conv),
        layers.MaxPooling2D(),
        layers.Conv2D(hp_conv_filters_start * 2, 3, activation='relu', padding="same",
                      kernel_initializer=keras.initializers.HeUniform(seed=weight_init_seed)),
        layers.Dropout(hp_drop_conv),
        layers.MaxPooling2D(),
        layers.Conv2D(hp_conv_filters_start * 4, 3, activation='relu', padding="same",
                      kernel_initializer=keras.initializers.HeUniform(seed=weight_init_seed)),
        layers.Dropout(hp_drop_conv),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(hp_dense_units, activation='relu',
                     kernel_initializer=keras.initializers.HeUniform(seed=weight_init_seed)),
        layers.Dropout(hp_drop_dense),
        layers.Dense(num_classes, kernel_initializer=keras.initializers.glorot_uniform(seed=weight_init_seed)),
        layers.Activation(keras.activations.sigmoid))
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[keras.metrics.CategoricalAccuracy(),
                           keras.metrics.AUC(multi_label=True),
                           keras.metrics.Recall(),
                           keras.metrics.Precision(),
                           keras.metrics.TopKCategoricalAccuracy(k=2, name="top 2 accuracy"),
                           keras.metrics.TopKCategoricalAccuracy(k=3, name="top 3 accuracy"),
                           keras.metrics.TopKCategoricalAccuracy(k=4, name="top 4 accuracy"),
                           keras.metrics.TopKCategoricalAccuracy(k=5, name="top 5 accuracy")])

    return model


def tune_params(train_ds, val_ds):
    tuner = kt.Hyperband(build_model,
                         objective=kt.Objective("val_categorical_accuracy", direction="max"),
                         max_epochs=100,
                         factor=3,
                         seed=15,
                         directory='tune_logs',
                         project_name='tune_lr')

    # Start search for parameters
    tuner.search(train_ds, validation_data=val_ds, epochs=800,
                 callbacks=[callback_earlystop(), tensorboard_setup("Tuning_search")],
                 verbose=2, workers=8)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps.values)

    # Find epoch when precision is best
    best_epoch = find_best_epoch(tuner, best_hps, train_ds, val_ds)
    # Retrain the model
    final_model = tuner.hypermodel.build(best_hps).fit(train_ds, epochs=best_epoch, validation_data=val_ds,
                                                       callbacks=[tensorboard_setup("final_model")], verbose=2,
                                                       workers=8, use_multiprocessing=True)
    final_model.save("cnn/model/final")


def find_best_epoch(tuner, best_hps, train_ds, val_ds):
    # Build the model with the optimal hyperparameters and train it on the data for 150 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_ds, epochs=150, validation_data=val_ds,
                        callbacks=[callback_earlystop(), tensorboard_setup("find_best_epoch")], verbose=2, workers=8,
                        use_multiprocessing=True)

    val_acc_per_epoch = history.history['val_precision']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    return best_epoch


def callback_earlystop():
    return keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


def tensorboard_setup(name):
    log_dir = "logs/fit/hypertune/" + name
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback
