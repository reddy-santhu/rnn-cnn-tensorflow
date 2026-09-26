"""
CS5720 Neural Network and Deep Learning - Home Assignment 2
Q5: Simplified AlexNet and a small ResNet-like model

Both models are defined and summarized. They are not trained.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

os.makedirs("outputs", exist_ok=True)


def build_alexnet(input_shape=(227, 227, 3), num_classes=10):
    """Layer list follows the assignment. Padding stays at Keras' default, valid.

    227x227 is the usual AlexNet input. With valid padding the maps stay
    large enough for the last 3x3 pool.
    """
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(96, (11, 11), strides=4, activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),
            tf.keras.layers.Conv2D(256, (5, 5), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),
            tf.keras.layers.Conv2D(384, (3, 3), activation="relu"),
            tf.keras.layers.Conv2D(384, (3, 3), activation="relu"),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="alexnet",
    )


def residual_block(input_tensor, filters):
    """Two 3x3 convolutions plus a skip connection.

    The assignment says to add the input to the block output before the
    activation. The first convolution uses ReLU. The second convolution is
    linear, the skip is added, and then ReLU is applied.

    padding='same' keeps the height and width so the skip can be added.
    A 1x1 convolution is used only when the channel counts differ.
    """
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(
        input_tensor
    )
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)

    if input_tensor.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding="same")(
            input_tensor
        )
    else:
        shortcut = input_tensor

    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.Activation("relu")(x)


def build_resnet(input_shape=(64, 64, 3), num_classes=10):
    """Stem convolution, two residual blocks, then flatten and two dense layers.

    The assignment does not set the class count. 10 matches the AlexNet head.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding="same", activation="relu")(
        inputs
    )
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="resnet_like")


def save_summary(model, filename):
    lines = []
    model.summary(print_fn=lines.append)
    text = "\n".join(lines)
    print(text)
    path = os.path.join("outputs", filename)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(f"\nSaved {path}\n")
    return text


def main():
    print("=" * 60)
    print("Task 1: AlexNet summary")
    print("=" * 60)
    alexnet = build_alexnet()
    save_summary(alexnet, "q5_alexnet_summary.txt")

    print("=" * 60)
    print("Task 2: ResNet-like summary")
    print("=" * 60)
    resnet = build_resnet()
    save_summary(resnet, "q5_resnet_summary.txt")


if __name__ == "__main__":
    main()
