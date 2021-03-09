from simple_cnn import train_cnn_model
from loader import load_data

if __name__ == '__main__':
    data_dir = "C:\\Users\\the_p\\Desktop\\d_set"
    train_ds, test_ds = load_data(data_dir)
    train_cnn_model(train_ds, test_ds)
