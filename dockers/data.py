import os
import tensorflow as tf
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from glob import glob


data_path = "flowers"
classnames = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
HEIGHT = 256
WIDTH = 256


def load():
    """
    Will Create X_train, X_test, y_train, y_test from the flower_photos file


    Returns
    X_train :
    y_train :
    X_test :
    y_test :

    """

    # Load the differents flowers photos
    daisy_path = glob(os.path.join(data_path, "daisy/*"))
    print(f"""Number of daisy : {len(daisy_path)}""")

    dandelion_path = glob(os.path.join(data_path, "dandelion/*"))
    print(f"""Number of dandelion : {len(dandelion_path)}""")

    roses_path = glob(os.path.join(data_path, "roses/*"))
    print(f"""Number of roses : {len(roses_path)}""")

    sunflowers_path = glob(os.path.join(data_path, "sunflowers/*"))
    print(f"""Number of sunflowers : {len(sunflowers_path)}""")

    tulips_path = glob(os.path.join(data_path, "tulips/*"))
    print(f"""Number of tulips : {len(tulips_path)}""")

    X_path = daisy_path + dandelion_path + roses_path + sunflowers_path + tulips_path
    # Need to resize here or else will get the "Can't convert non-rectangular Python sequence to Tensor." error during the from_tensor_slices
    X = [cv2.resize(cv2.imread(x, cv2.IMREAD_COLOR), (HEIGHT, WIDTH)) for x in X_path]

    y_daisy = np.zeros((len(daisy_path), 1), dtype=np.int8)
    y_dandelion = np.zeros((len(dandelion_path), 1), dtype=np.int8) + 1
    y_roses = np.zeros((len(roses_path), 1), dtype=np.int8) + 2
    y_sunflowers = np.zeros((len(sunflowers_path), 1), dtype=np.int8) + 3
    y_tulips = np.zeros((len(tulips_path), 1), dtype=np.int8) + 4

    y = np.concatenate([y_daisy, y_dandelion, y_roses, y_sunflowers, y_tulips], axis=0)
    y = y.astype(dtype=np.int16)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return (X_train, y_train), (X_test, y_test)


def preprocess(x, y, scale=False):
    x = tf.image.resize(x, (HEIGHT, WIDTH))
    x = tf.cast(x, dtype=tf.float32)
    if scale:
        x = x / 255.0  # Since we are going to use an EfficientNet
    return x, y


def tf_dataset(x, y, batch_size=32, repeat=False):
    y = tf.constant(y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load()
    # X_train, y_train = X_train[:64], y_train[:64]
    train_ds = tf_dataset(X_train, y_train)
    # test_ds = tf_dataset(X_test, y_test)
    # print(train_ds)
