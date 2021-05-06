from simple_model.simple_cnn import train_cnn_model, predict_sample_image
from loader import load_data, visualize_data
from datetime import timedelta
from image_processing import image_downloader, image_labeler
from suatap_paper import suatap_model
from tensorflow.python.client import device_lib
import time
import argparse


def gather_images():
    time_before = time.time()
    image_downloader.download_images()
    stop_timer(time_before, "Time spent on gathering URL links: ")


def download_images():
    time_before = time.time()
    image_downloader.download_images_from_file()
    stop_timer(time_before, "Time spent on downloading images")


def label_images():
    time_before = time.time()
    image_labeler.label_images()
    stop_timer(time_before, "Time spent on labeling images")


def load_data_from_directory(directory):
    time_before = time.time()
    train_dataset, test_dataset = load_data(directory)
    stop_timer(time_before, "Spent on loading images")
    return train_dataset, test_dataset


def train_model(train_set, test_set):
    time_before = time.time()
    train_cnn_model(train_set, test_set)
    stop_timer(time_before, "Spent on training Simple_cnn")


def train_suatap_model(train_ds, val_ds):
    model = suatap_model.cnn_setup(1, 3, len(train_ds.class_indices.items()), (32, 32, 3))
    time_start = time.time()
    suatap_model.train_model(model, train_ds, val_ds)
    stop_timer(time_start, "Spent on training Suatap et. al.")


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
    parser.add_argument("-dl", "--download", action="store_true", help="Download and label dataset")
    parser.add_argument("--skip_data", action="store_true", help="Skip loading dataset")
    parser.add_argument("-simple", "--simple_cnn", action="store_true", help="Train simple_cnn model")
    parser.add_argument("--predict_simple", action="store_true", help="Predict labels for simple_cnn model")
    parser.add_argument("--suatap", action="store_true", help="Train Suatap model")
    parser.add_argument("--devices", action="store_true", help="Show available (GPU) devices")
    parser.add_argument("--predict_suatap", action="store_true", help="Predict labels for Suatap model")
    parser.add_argument("--visualize", action="store_true", help="Visualize 9 images and their labels")

    cli_args = parser.parse_args()

    # Gather and Download images
    if cli_args.download:
        gather_images()
        download_images()

        # Label images
        label_images()

    # Load data
    if not cli_args.skip_data:
        train_ds, test_ds = load_data_from_directory("../resources/all_images")

    # Visualize 9 images from the training set
    if cli_args.visualize:
        visualize_data(train_ds, 9)

    # Train simple model
    if cli_args.simple_cnn:
        train_model(train_ds, test_ds)

    # Predict on sample_image based on labels from the training set
    if cli_args.predict_simple:
        img_labels = []
        for k, v in train_ds.class_indices.items():
            img_labels.append(k)
        predict_sample_image(img_labels)

    # Train model from Suatap paper
    if cli_args.suatap:
        train_suatap_model(train_ds, test_ds)

    # List available devices (CPU/GPU)
    if cli_args.devices:
        show_devices()

    # Predict from Suatap
    if cli_args.predict_suatap:
        suatap_model.inference_mode("")
