from cgi import test
import tensorflow as tf
from data import *
from model import base_model, fine_tuned_model, other_model
import wandb
from wandb.keras import WandbCallback

CHECKPOINT_PATH = "./finetuned_model/"

wandb.init(
    project="base_model",
    entity="yuuuugo",
)
wandb.config.epochs = 5
wandb.config.batch_size = 32
wandb.config.learning_rate = 1e-3
wandb.config.architecture = "efficientnet"


if __name__ == "__main__":

    (X_train, y_train), (X_test, y_test) = load()
    X_train, y_train = X_train, y_train
    X_test, y_test = X_test, y_test
    train_ds = tf_dataset(X_train, y_train, batch_size=32)
    test_ds = tf_dataset(X_test, y_test, batch_size=32)

    """ tf.keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_sparse_categorical_accuracy",
            save_best_only=True,
            save_weights_only=True,
        ), """
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5),
        WandbCallback(
            monitor="val_loss",
            verbose=0,
        ),
    ]
    model = base_model(5)
    # model = other_model()
    # model = fine_tuned_model(5, 5)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=wandb.config.epochs,
        batch_size=wandb.config.batch_size,
        steps_per_epoch=len(train_ds),
        callbacks=callbacks,
    )
    model.save("base_model.h5")
