from simple_cnn import train_cnn_model, predict_sample_image
from loader import load_data
import time
import image_downloader


def download_images():
    time_before = time.time()
    image_downloader.download_images()
    print("Time spent on downloading images: " + str(time.time() - time_before))


def load_data_from_directory(directory):
    time_before = time.time()
    train_dataset, test_dataset = load_data(directory)
    print("Time spent on loading images: " + str(time.time() - time_before))
    return train_dataset, test_dataset


def train_model(train_set, test_set):
    time_before = time.time()
    train_cnn_model(train_set, test_set)
    print("Time spent on training: " + str(time.time() - time_before))


if __name__ == '__main__':
    # Download images
    # download_images()

    # Load data
    train_ds, test_ds = load_data_from_directory("genres")

    # Visualize 9 images from the training set
    # visualize_data(train_set, 9)

    # Predict on sample_image based on labels from the training set
    class_names = train_ds.class_names
    predict_sample_image(class_names)

    # Train model
    # train_model(train_ds, test_ds)
