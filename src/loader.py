import numpy as np
import os
# PILLOW lib
import PIL
import PIL.Image
import tensorflow as tf
# tensorflow-datasets lib
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt


def load_data(data_path, validation_percent):
    """
    Loads and labels images from a directory. Splits images into training and validation sets.

    :param validation_percent: Percentage saved for validation of model
    :param data_path: The path to dataset directory
    :return: A tuple consisting of 2 tf.data.Datset objects.The training and validation datasets respectively.
    """
    # Re-create path into object-oriented system
    data_dir = pathlib.Path(data_path)

    batch_size = 32
    img_height = 180
    img_width = 180

    # Classes defined by directory structure
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_percent,
        subset="training",
        seed=123,
        shuffle=True,
        label_mode='binary',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_percent,
        subset="validation",
        seed=123,
        shuffle=True,
        label_mode='binary',
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds, validation_ds


def visualize_data(dataset, img_count):
    """
    Visualizes part of dataset in plot

    :param dataset: The dataset to be visualized
    :param img_count: Number of images to show
    """

    class_names = dataset.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in dataset:
        for i in range(img_count):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            # For the current index, get the label value (0.0 or 1.0)
            label_value = int(labels[i])
            # Use label value to lookup proper class name (Original, Tampered)
            plt.title(class_names[label_value])
            plt.axis("off")

    plt.show()


train_set, valid_set = load_data("C:\\Users\\the_p\\Desktop\\d_set")
visualize_data(train_set, 9)
