import pathlib
import ast
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


def load_data(data_path, validation_percent=0.2, batch_size=32, img_height=32, img_width=32):
    """
    Loads and labels images from a directory. Splits images into training and validation sets.

    :param img_width: Width of images after resizing
    :param img_height: Height of images after resizing
    :param batch_size: The size of batches of data
    :param validation_percent: Percentage saved for validation of model
    :param data_path: The path to dataset directory
    :return: A tuple consisting of 2 tf.data.Dataset objects.The training and validation datasets respectively.
    """
    # Re-create path into object-oriented system
    data_dir = pathlib.Path(data_path)

    # Classes defined by directory structure
    # train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     data_dir,
    #     validation_split=validation_percent,
    #     subset="training",
    #     seed=123,
    #     shuffle=True,
    #     label_mode='int',
    #     labels='inferred',
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size)

    generator = tf.keras.preprocessing.image.ImageDataGenerator()

    column_labeled_images = ["filename", "labels"]
    df_labeled_images = pd.read_csv("image_labels.txt", sep="|", names=column_labeled_images)
    df_labeled_images["labels"] = df_labeled_images["labels"].apply(lambda x: ast.literal_eval(x))

    train_ds = generator.flow_from_dataframe(
        df_labeled_images,
        data_dir,
        x_col="filename",
        y_col="labels",
        target_size=(img_height, img_width),
        subset="training",
        seed=15,
        shuffle=True,
        data_format=None,
        batch_size=batch_size
    )

    # validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     data_dir,
    #     validation_split=validation_percent,
    #     subset="validation",
    #     seed=123,
    #     shuffle=True,
    #     label_mode='int',
    #     labels='inferred',
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size)

    validation_ds = generator.flow_from_dataframe(
        df_labeled_images,
        data_dir,
        x_col="filename",
        y_col="labels",
        target_size=(img_height, img_width),
        subset="validation",
        seed=15,
        shuffle=True,
        data_format=None,
        batch_size=batch_size
    )

    return train_ds, validation_ds


def visualize_data(dataset, img_count):
    """
    Visualizes part of dataset in plot

    :param dataset: The dataset to be visualized
    :param img_count: Number of images to show
    """

    plt.figure(figsize=(10, 10))

    images, labels = next(iter(dataset))
    for i in range(img_count):
        # Get labels for current image
        img_labels = []
        inv_labels = {v: k for k, v in dataset.class_indices.items()}
        it = np.nditer(labels[i], flags=['f_index'])
        for val in it:
            if val > 0:
                img_labels.append(inv_labels[it.index])
        label_value = img_labels
        # Show images and labels
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(str(label_value))
        plt.axis("off")

    plt.show()
