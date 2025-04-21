import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_cnn(num_layers, input_shape=(224, 224, 3), num_classes=7):
    model = keras.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    for _ in range(num_layers):
        model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
