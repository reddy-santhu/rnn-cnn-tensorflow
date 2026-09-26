"""
CS5720 Neural Network and Deep Learning - Home Assignment 2
Q2: Sentiment Classification Using RNN

Train an LSTM on IMDB reviews and report a confusion matrix plus
precision, recall, and F1 from sklearn.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

tf.random.set_seed(42)
np.random.seed(42)

os.makedirs("outputs", exist_ok=True)

VOCAB_SIZE = 10000
MAXLEN = 150
# A subset keeps training time reasonable. The sequences are still real IMDB reviews.
TRAIN_SAMPLES = 6000
TEST_SAMPLES = 2000
EMBED_DIM = 32
LSTM_UNITS = 32
BATCH_SIZE = 64
EPOCHS = 3


def build_model():
    """Padded word ids -> embedding -> LSTM -> one sigmoid unit (positive vs negative)."""
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(MAXLEN,)),
            tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM),
            tf.keras.layers.LSTM(LSTM_UNITS),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("=" * 60)
    print("Q2: IMDB sentiment classification")
    print("=" * 60)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=VOCAB_SIZE
    )
    # Reviews have different lengths. Padding / truncating makes one tensor.
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAXLEN)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAXLEN)

    x_train = x_train[:TRAIN_SAMPLES]
    y_train = y_train[:TRAIN_SAMPLES]
    x_test = x_test[:TEST_SAMPLES]
    y_test = y_test[:TEST_SAMPLES]
    print(f"Train shape: {x_train.shape}")
    print(f"Test shape : {x_test.shape}")

    model = build_model()
    model.summary()
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )

    probabilities = model.predict(x_test, verbose=0).reshape(-1)
    # 0.5 is the usual cutoff for the required classification report.
    predictions = (probabilities >= 0.5).astype(int)

    matrix = confusion_matrix(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=["negative", "positive"],
        digits=4,
    )
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print("\nConfusion matrix")
    print("rows = true label, columns = predicted label")
    print("order: negative, positive")
    print(matrix)
    print("\nClassification report")
    print(report)
    print(f"Test loss    : {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    lines = [
        "Q2 sentiment classification",
        "Confusion matrix (rows = true, columns = predicted; order = negative, positive)",
        str(matrix),
        "",
        "Classification report (threshold = 0.5)",
        report.rstrip(),
        "",
        f"Test loss    : {loss:.4f}",
        f"Test accuracy: {accuracy:.4f}",
        "",
    ]
    out_path = os.path.join("outputs", "q2_classification_report.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
