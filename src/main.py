from simple_cnn import train_cnn_model
from loader import load_data
import time
import image_downloader

if __name__ == '__main__':
    # entry point, run with image example or train model
    # run_example()
    # train_set, valid_set = load_data("genres")
    # visualize_data(train_set, 9)

    # Download images
    time_before = time.time()
    image_downloader.download_images()
    print("Time spent on downloading images: " + str(time.time() - time_before))

    # Load data
    time_before = time.time()
    data_dir = "genres"
    train_ds, test_ds = load_data(data_dir)
    print("Time spent on loading images: " + str(time.time() - time_before))

    # Train model
    time_before = time.time()
    train_cnn_model(train_ds, test_ds)
    print("Time spent on training: " + str(time.time() - time_before))
