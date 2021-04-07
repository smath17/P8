from simple_model.simple_cnn import train_cnn_model
from loader import load_data
import time
from datetime import timedelta
from image_processing import image_downloader, image_labeler
from suatap_paper import suatap_model
from tensorflow.python.client import device_lib


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
    model = suatap_model.cnn_setup(1, 3, 27, (32, 32, 3))
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
    # Gather and Download images
    gather_images()
    download_images()

    # Label images
    label_images()

    # Load data
    # train_ds, test_ds = load_data_from_directory("genres")

    # Visualize 9 images from the training set
    # visualize_data(train_set, 9)

    # Predict on sample_image based on labels from the training set
    # class_names = train_ds.class_names
    # predict_sample_image(class_names)

    # Train simple model
    # train_model(train_ds, test_ds)

    # Train model from Suatap paper
    # train_suatap_model(train_ds, test_ds)

    # List available devices (CPU/GPU)
    # show_devices()
