import argparse
import time
from datetime import timedelta

# Disable tensorflow logging
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

from tensorflow.python.client import device_lib

import hypertune_simple
from binary_model.basic_cnn import train, predict
from image_processing import image_downloader, image_labeler
from image_processing.image_labeler import label_images_with_rest
from loader import load_data, visualize_data, load_test_data
from simple_model.simple_cnn import train_cnn_model, predict_sample_image, simple_evaluate
from suatap_paper import suatap_model


def gather_images():
    time_before = time.time()
    image_downloader.gather_images()
    stop_timer(time_before, "Time spent on gathering URL links: ")


def download_images():
    time_before = time.time()
    image_downloader.download_images_from_file()
    stop_timer(time_before, "Time spent on downloading images")


def label_images():
    time_before = time.time()
    image_labeler.label_images()
    stop_timer(time_before, "Time spent on labeling images")


def load_data_from_directory(directory, sampling, rest_label):
    time_before = time.time()
    train_dataset, validation_dataset = load_data(directory, sampling, rest_label, img_height=256, img_width=256, batch_size=256)
    stop_timer(time_before, "Spent on loading images")
    return train_dataset, validation_dataset


def evaluate_simple_cnn(directory, sampling, rest_label, name):
    # Make sure images have same size as when trained
    test_data = load_test_data(directory, sampling, rest_label, img_width=256, img_height=256, batch_size=256)
    simple_evaluate(test_data, name)


def train_model(train_set, test_set, model_name="standard"):
    time_before = time.time()
    train_cnn_model(train_set, test_set, model_name)
    stop_timer(time_before, "Spent on training Simple_cnn")


def train_suatap_model(train_ds, val_ds):
    model = suatap_model.cnn_setup(1, 5, len(train_ds.class_indices.items()), (256, 256, 3))
    time_start = time.time()
    suatap_model.train_model(model, train_ds, val_ds)
    stop_timer(time_start, "Spent on training Suatap et. al.")


def tune_simple(train_ds, val_ds):
    time_start = time.time()
    hypertune_simple.tune_params(train_ds, val_ds)
    stop_timer(time_start, "Spent on tuning learning rate.")


def stop_timer(time_start, text="Spent on training"):
    time_elapsed = time.time() - time_start
    print(str(timedelta(seconds=time_elapsed)) + " " + text)


def show_devices():
    """
    Lists all devices available to Tensorflow/Keras
    """
    print(device_lib.list_local_devices())


if __name__ == '__main__':
    # Initialize CLI arguments
    # -h for help
    parser = argparse.ArgumentParser()
    parser.add_argument("-gather", "--gather_urls", action="store_true", help="Prepare dataset for downloading images")
    parser.add_argument("-dl", "--download", action="store_true", help="Download dataset")
    parser.add_argument("-label", action="store_true", help="Label dataset")
    parser.add_argument("--rest_label", action="store_true", help="Use 5 largest classes and 1 class for the rest.")
    parser.add_argument("--skip_data", action="store_true", help="Skip loading dataset")
    parser.add_argument("--sample", action="store_true", help="Attempt to sample the dataset")
    parser.add_argument("-simple", "--simple_cnn", action="store_true", help="Train simple_cnn model")
    parser.add_argument("-binary", "--binary_cnn", action="store_true", help="Train binary model")
    parser.add_argument("--predict_simple", action="store_true", help="Predict labels for simple_cnn model")
    parser.add_argument("--predict_binary", action="store_true", help="Predict labels for binary model")
    parser.add_argument("--suatap", action="store_true", help="Train Suatap model")
    parser.add_argument("--devices", action="store_true", help="Show available (GPU) devices")
    parser.add_argument("--predict_suatap", action="store_true", help="Predict labels for Suatap model")
    parser.add_argument("--visualize", action="store_true", help="Visualize 9 images and their labels")
    parser.add_argument("--steps_per_epoch", type=int, default=0, help="Amount of steps per training epoch")
    parser.add_argument("--epoch_count", type=int, default=0, help="Amount of epochs during training")
    parser.add_argument("--evaluate_simple", action="store_true")
    parser.add_argument('-name', action='store', type=str, help='The name of the model.')
    parser.add_argument('--tune_params', action='store_true')

    cli_args = parser.parse_args()

    # Gather image URLs
    if cli_args.gather_urls:
        image_labeler.label_apps()
        gather_images()

    # Download images
    if cli_args.download:
        download_images()

    # Label images
    if cli_args.label:
        # Label images
        if cli_args.rest_label:
            label_images_with_rest()
        else:
            label_images()

    # Load data
    if not cli_args.skip_data:
        train_ds, val_ds = load_data_from_directory("../resources/all_images", cli_args.sample, cli_args.rest_label)

    # Visualize 9 images from the training set
    if cli_args.visualize and not cli_args.skip_data:
        visualize_data(train_ds, 9)

    # Train simple model
    if cli_args.simple_cnn and not cli_args.skip_data:
        if cli_args.name:
            train_model(train_ds, val_ds, cli_args.name)
        else:
            train_model(train_ds, val_ds)

    # Predict on sample_image based on labels from the training set
    if cli_args.predict_simple:
        img_labels = []
        for k, v in train_ds.class_indices.items():
            img_labels.append(k)
        predict_sample_image(img_labels)

    # Train model from Suatap paper
    if cli_args.suatap:
        train_suatap_model(train_ds, val_ds)

    # Train binary model
    if cli_args.binary_cnn:
        train(cli_args.steps_per_epoch, cli_args.epoch_count)

    # Predict label using binary model
    if cli_args.predict_binary:
        predict()

    # List available devices (CPU/GPU)
    if cli_args.devices:
        show_devices()

    # Predict from Suatap
    if cli_args.predict_suatap:
        suatap_model.inference_mode("")

    if cli_args.evaluate_simple:
        evaluate_simple_cnn("../resources/all_images", cli_args.sample, cli_args.rest_label, cli_args.name)

    if cli_args.tune_params:
        tune_simple(train_ds, val_ds)
