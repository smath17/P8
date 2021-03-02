import numpy as np
import os
# PILLOW lib
import PIL
import PIL.Image
import tensorflow as tf
# tensorflow-datasets lib
import tensorflow_datasets as tfds
import pathlib


def load_data():
    dataset_path = "C:\\Users\\the_p\\Desktop\\images"
    # Re-create path into object-oriented system
    data_dir = pathlib.Path(dataset_path)
    image_count = len(list(data_dir.glob('*.tif')))
    print(image_count)

    images = list(data_dir.glob('*.tif'))
    first = PIL.Image.open(str(images[0]))

    # Show picture, cannot show .tif
    # PIL.Image.Image.show(first)

    batch_size = 32
    img_height = 421
    img_width = 424

    # TODO: tif not supported, convert to jpeg, change directory structure
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)


load_data()
