import ast
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def __prepare_dataframe(sampling, rest_label, rand_seed):
    """
    Prepares the dataframe by loading csv data in regards to resampling and labels.
    Splits the data into train and test sets.

    :param rest_label: Boolean determining whether we use 5+1 labelling.
    :param sampling: Boolean determining whether we should try to resample the data.
    :return: A tuple consisting of 2 pandas dataframes. The training and test datasets respectively.
    """

    column_labeled_images = ["filename", "labels"]
    if rest_label:
        df_labeled_images = pd.read_csv("image_labels_2.txt", sep="|", names=column_labeled_images)
    else:
        df_labeled_images = pd.read_csv("image_labels.txt", sep="|", names=column_labeled_images)

    # 10% of data is saved for evaluation
    train_val_ds, test_ds = train_test_split(df_labeled_images, test_size=0.1, random_state=rand_seed)

    return train_val_ds, test_ds


def load_data(data_path, sampling, rest_label, validation_percent=0.2, batch_size=64, img_height=256, img_width=256,
              rand_seed=15):
    """
       Loads the dataframe containing train/validation data and splits into separate sets.

       :param rand_seed: The seed used to reproduce random sets
       :param rest_label: Boolean determining whether we use 5+1 labelling.
       :param sampling: Boolean determining whether we should try to resample the data.
       :param img_width: Width of images after resizing
       :param img_height: Height of images after resizing
       :param batch_size: The size of batches of data
       :param validation_percent: Percentage saved for validation of model
       :param data_path: The path to dataset directory
       :return: A tuple consisting of 2 dataframe objects.The training and validation datasets respectively.
       """

    train_val_ds = __prepare_dataframe(sampling, rest_label, rand_seed)[0]
    # Make a copy in order to manipulate train_ds without affecting validation_ds
    copy_ds = train_val_ds.copy()

    generator = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=validation_percent)

    # Labels with > 50.000 entries
    high_data_list = ["casual", "indie", "adventure", "action", "strategy"]
    low_data_list = []

    with open("../resources/tags.txt") as file:
        for line in file:
            genre = line.strip("\n")
            if genre not in low_data_list and genre not in high_data_list:
                low_data_list.append(genre)

    if sampling:
        print("Attempting to resample dataset...")
        if rest_label:
            train_val_ds = undersample_rest_label(train_val_ds, high_data_list)
        else:
            train_val_ds = oversample_low_data(train_val_ds, low_data_list, high_data_list)
        print("Done resampling the dataset.")

    print_data_distribution(train_val_ds, low_data_list, high_data_list, rest_label)

    # Convert string to lists to be used in dataframe
    train_val_ds["labels"] = train_val_ds["labels"].apply(lambda x: ast.literal_eval(x))
    copy_ds["labels"] = copy_ds["labels"].apply(lambda x: ast.literal_eval(x))

    train_ds = generator.flow_from_dataframe(
        train_val_ds,
        data_path,
        x_col="filename",
        y_col="labels",
        target_size=(img_height, img_width),
        subset="training",
        seed=rand_seed,
        shuffle=True,
        data_format=None,
        batch_size=batch_size
    )

    validation_ds = generator.flow_from_dataframe(
        copy_ds,
        data_path,
        x_col="filename",
        y_col="labels",
        target_size=(img_height, img_width),
        subset="validation",
        seed=rand_seed,
        shuffle=True,
        data_format=None,
        batch_size=batch_size
    )

    return train_ds, validation_ds


def load_test_data(data_path, sampling, rest_label, rand_seed=15, batch_size=64, img_height=256, img_width=256):
    test_df = __prepare_dataframe(sampling, rest_label, rand_seed)[1]

    # Convert string to lists to be used in dataframe
    test_df["labels"] = test_df["labels"].apply(lambda x: ast.literal_eval(x))

    generator = tf.keras.preprocessing.image.ImageDataGenerator()

    return generator.flow_from_dataframe(
        test_df,
        data_path,
        x_col="filename",
        y_col="labels",
        target_size=(img_height, img_width),
        batch_size=batch_size
    )


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


def print_data_distribution(df_labeled_images, low_data, high_data, rest_label):
    print("Genre/Tag, # of screenshots")

    if rest_label:
        dataframe = df_labeled_images.loc[
            df_labeled_images["labels"].map(lambda x: "rest" in x)
        ]
        print("rest" + ", " + str(len(dataframe)))
    else:
        for genre in low_data:
            dataframe = df_labeled_images.loc[
                df_labeled_images["labels"].map(lambda x: genre in x)
            ]
            print(genre + ", " + str(len(dataframe)))

    for genre in high_data:
        dataframe = df_labeled_images.loc[
            df_labeled_images["labels"].map(lambda x: genre in x)
        ]
        print(genre + ", " + str(len(dataframe)))

    print("Total amount of samples: " + str(len(df_labeled_images)))


def oversample_low_data(df_labeled_images, low_data_list, high_data_list):
    high_dataframes = []
    for oversampled in high_data_list:
        # Dataframe consisting of action labelled images.
        high_dataframes.append(df_labeled_images.loc[
                                   df_labeled_images["labels"].map(lambda x: oversampled in x)
                               ])

    for _ in range(0, 3):
        for genre in low_data_list:
            counter = 0
            while True:
                # Dataframe consisting of fighting labelled images.
                low_dataframe = df_labeled_images.loc[
                    df_labeled_images["labels"].map(lambda x: genre in x)
                ]

                # If we reached a good distribution
                if 20000 <= len(low_dataframe):
                    break

                # This does not affect image_labels.txt so this is reset every run.
                # Oversample or undersample label
                if len(low_dataframe) <= 20000:
                    low_dataframe = low_dataframe.sample(20000 - len(low_dataframe), replace=True)  # Oversample

                    # Combine sampling with dataset
                    df_labeled_images = df_labeled_images.append(low_dataframe)
                else:
                    low_dataframe = low_dataframe.sample(20000)  # Undersample

                    # Remove previous labelled images
                    df_labeled_images = df_labeled_images.loc[
                        df_labeled_images["labels"].map(lambda x: genre not in x)
                    ]
                    # Add new labelled images
                    df_labeled_images = df_labeled_images.append(low_dataframe)

                # Break early as some classes cannot be resampled
                # due to them being so closely related to any of the high data classes.
                # Whenever we remove the high data class samples, the new added low data samples are removed also.
                if counter > 30:
                    break

                # Resample. This potentially removes some of the low data.
                # Thus we need to iterate this whole process.
                df_labeled_images = resample_high_data_labels(df_labeled_images, high_data_list, high_dataframes)

                counter += 1

    return df_labeled_images


def resample_high_data_labels(df_labeled_images, high_data_list, high_dataframes):
    counter = 0
    for high_dataframe in high_dataframes:
        high_dataframes[counter] = high_dataframe.sample(20000)
        counter += 1

    # Remove previous action labelled images
    df_labeled_images = df_labeled_images.loc[
        df_labeled_images["labels"].map(lambda x: high_data_list[0] not in x and
                                                  high_data_list[1] not in x and
                                                  high_data_list[2] not in x and
                                                  high_data_list[3] not in x and
                                                  high_data_list[4] not in x)
    ]

    # Add new resampled action labelled images
    for high_dataframe in high_dataframes:
        df_labeled_images = df_labeled_images.append(high_dataframe)

    return df_labeled_images


def print_length_of_frame(df_labeled_images, genre):
    low_dataframe = df_labeled_images.loc[
        df_labeled_images["labels"].map(lambda x: genre in x)
    ]
    print(genre + ": " + str(len(low_dataframe)))


def undersample_rest_label(df_labeled_images, high_data_list):
    high_dataframes_size = 0
    for oversampled in high_data_list:
        # Dataframe consisting of high_data labelled images.
        high_dataframes_size += len(df_labeled_images.loc[
                                        df_labeled_images["labels"].map(lambda x: oversampled in x)
                                    ])

    average_dataframe_size = math.ceil(high_dataframes_size / len(high_data_list))
    rest_dataframe = df_labeled_images.loc[
        df_labeled_images["labels"].map(lambda x: "rest" in x)
    ]
    rest_dataframe = rest_dataframe.sample(average_dataframe_size)

    # Remove previous labelled images
    df_labeled_images = df_labeled_images.loc[
        df_labeled_images["labels"].map(lambda x: "rest" not in x)
    ]
    # Add new labelled images
    df_labeled_images = df_labeled_images.append(rest_dataframe)

    return df_labeled_images
