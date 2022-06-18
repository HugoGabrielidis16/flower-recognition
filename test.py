from multiprocessing.spawn import prepare
import tensorflow as tf
from random import randint, random
from model import base_model, other_model, fine_tuned_model
import cv2
import numpy as np
from data import classnames
from glob import glob
import os
import matplotlib.pyplot as plt


HEIGHT = 256
WIDTH = 256
data_path = "flowers"


def load_model(name):
    """
    Load one the model & setting the weights we found after training
    """

    if name == "finetuned":
        """model = fine_tuned_model()
        model.load_weights("finetuned_model/cp.cpkt")"""
        model = tf.keras.models.load_model("finetuned_model.h5")
    elif name == "base":
        model = base_model()
        model.load_weights("base_model/cp.cpkt")
    elif name == "other":
        model = other_model()
        model.load_weights("other_model/cp.cpkt")
    return model


def prepare_image(image):
    image = tf.image.resize(image, (HEIGHT, WIDTH))
    image = tf.cast(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)  # the model takes batch
    return image


def inferance(model, image):
    prediction = model.predict(image)
    return classnames[np.argmax(prediction)]


def random_sample():
    daisy_path = glob(os.path.join(data_path, "daisy/*"))
    dandelion_path = glob(os.path.join(data_path, "dandelion/*"))
    roses_path = glob(os.path.join(data_path, "roses/*"))
    sunflowers_path = glob(os.path.join(data_path, "sunflowers/*"))
    tulips_path = glob(os.path.join(data_path, "tulips/*"))
    X_path = daisy_path + dandelion_path + roses_path + sunflowers_path + tulips_path

    y_daisy = np.zeros((len(daisy_path), 1), dtype=np.int8)
    y_dandelion = np.zeros((len(dandelion_path), 1), dtype=np.int8) + 1
    y_roses = np.zeros((len(roses_path), 1), dtype=np.int8) + 2
    y_sunflowers = np.zeros((len(sunflowers_path), 1), dtype=np.int8) + 3
    y_tulips = np.zeros((len(tulips_path), 1), dtype=np.int8) + 4
    y = np.concatenate(
        [y_daisy, y_dandelion, y_roses, y_sunflowers, y_tulips],
        axis=0,
    )

    random_n = randint(0, len(X_path))
    index = y[random_n][0]
    y_true = classnames[index]
    X = cv2.imread(X_path[random_n], cv2.IMREAD_COLOR)

    return X, y_true


if __name__ == "__main__":
    # Load the differents flowers photos

    X_sample, y_sample = random_sample()
    X_original = X_sample
    X_sample = prepare_image(X_sample)

    model = load_model("finetuned")
    prediction = model.predict(X_sample)

    class_predicted = classnames[np.argmax(prediction)]
    print(y_sample, class_predicted)
    plt.imshow(X_original)
    plt.title(f""" Predicted label : {class_predicted}, actual label : {y_sample}""")
    plt.show()
