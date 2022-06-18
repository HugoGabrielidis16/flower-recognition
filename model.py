import tensorflow as tf
from tensorflow.keras import layers


def base_model(num_class):

    """
    Transfer Learning with BERTs
    """
    EfficientNetB0 = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet"
    )
    EfficientNetB0.trainable = False

    inputs = layers.Input(shape=(256, 256, 3))
    x = EfficientNetB0(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_class, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def fine_tuned_model(num_class, n):
    """
    Fine tuned model with image augmentation & some unfreezed layers of EfficientNet
    """
    img_augmentation = tf.keras.Sequential(
        [
            layers.RandomRotation(factor=0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomFlip(),
            layers.RandomContrast(factor=0.1),
        ],
        name="img_augmentation",
    )

    EfficientNetB0 = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", drop_connect_rate=0.4
    )
    EfficientNetB0.trainable = True

    # Freeze the first n layers
    for layer in EfficientNetB0.layers[:-n]:
        layer.trainable = False

    # Freeze BatchNorm layers
    for layer in EfficientNetB0.layers[-n:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    inputs = layers.Input(shape=(256, 256, 3))
    x = img_augmentation(inputs)
    x = EfficientNetB0(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4, name="top_dropout")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_class, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def other_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            padding="Same",
            activation="relu",
            input_shape=(150, 150, 3),
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(
        layers.Conv2D(filters=64, kernel_size=(3, 3), padding="Same", activation="relu")
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(
        layers.Conv2D(filters=96, kernel_size=(3, 3), padding="Same", activation="relu")
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(
        layers.Conv2D(filters=96, kernel_size=(3, 3), padding="Same", activation="relu")
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(5, activation="softmax"))

    return model


if __name__ == "__main__":
    model = fine_tuned_model(5, 20)
    model.summary()
