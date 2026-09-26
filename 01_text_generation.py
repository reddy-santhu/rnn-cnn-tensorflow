"""
CS5720 Neural Network and Deep Learning - Home Assignment 2
Q1: Implementing an RNN for Text Generation

Train a character-level LSTM on Shakespeare and sample the next character
one at a time. Temperature scaling is applied to the logits before softmax.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

# Same characters every rerun, so the sample text is reproducible.
tf.random.set_seed(42)
np.random.seed(42)

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

SHAKESPEARE_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)
# A slice of the file is enough to show next-character training on a laptop.
CORPUS_CHARS = 20000
SEQ_LENGTH = 40
STEP = 3
EMBED_DIM = 64
LSTM_UNITS = 128
BATCH_SIZE = 128
EPOCHS = 20
GENERATE_LENGTH = 280
TEMPERATURES = (0.5, 1.0, 1.5)


def load_text():
    """Download the Shakespeare file once and keep it under data/."""
    path = tf.keras.utils.get_file(
        "shakespeare.txt",
        origin=SHAKESPEARE_URL,
        cache_dir=".",
        cache_subdir="data",
    )
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    return text[:CORPUS_CHARS]


def build_sequences(encoded, seq_length, step):
    """Each input is seq_length characters. The target is the next one."""
    inputs = []
    targets = []
    last_start = len(encoded) - seq_length
    for start in range(0, last_start, step):
        inputs.append(encoded[start : start + seq_length])
        targets.append(encoded[start + seq_length])
    return np.asarray(inputs), np.asarray(targets)


def build_model(vocab_size, seq_length):
    """Embedding turns character ids into vectors. LSTM predicts the next id.

    The dense layer has no softmax. Loss uses from_logits=True so temperature
    can rescale the raw scores before sampling.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(seq_length,)),
            tf.keras.layers.Embedding(vocab_size, EMBED_DIM),
            tf.keras.layers.LSTM(LSTM_UNITS),
            tf.keras.layers.Dense(vocab_size),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model


def sample_next_id(logits, temperature):
    """Divide logits by T, then softmax. Lower T is sharper. Higher T is flatter."""
    scaled = logits / temperature
    probs = tf.nn.softmax(scaled).numpy()
    return np.random.choice(len(probs), p=probs)


def generate_text(model, seed, length, temperature, char_to_id, id_to_char):
    """Slide a window of SEQ_LENGTH characters and append one sampled character."""
    window = seed
    pieces = [seed]
    for _ in range(length):
        ids = np.array([[char_to_id[ch] for ch in window]])
        logits = model(ids, training=False).numpy()[0]
        next_id = sample_next_id(logits, temperature)
        next_char = id_to_char[next_id]
        pieces.append(next_char)
        window = window[1:] + next_char
    return "".join(pieces)


def main():
    print("=" * 60)
    print("Q1: LSTM character text generation")
    print("=" * 60)

    text = load_text()
    chars = sorted(set(text))
    char_to_id = {ch: index for index, ch in enumerate(chars)}
    id_to_char = {index: ch for ch, index in char_to_id.items()}
    encoded = np.array([char_to_id[ch] for ch in text], dtype=np.int32)

    x_train, y_train = build_sequences(encoded, SEQ_LENGTH, STEP)
    print(f"Corpus characters : {len(text)}")
    print(f"Vocabulary size   : {len(chars)}")
    print(f"Training pairs    : {len(x_train)}")

    model = build_model(len(chars), SEQ_LENGTH)
    model.summary()
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # Seed must be exactly SEQ_LENGTH characters that exist in the vocabulary.
    seed = text[200 : 200 + SEQ_LENGTH]
    lines = [
        "Q1 text generation",
        f"Seed: {seed!r}",
        "",
    ]
    for temperature in TEMPERATURES:
        # Reseed so the only thing that changes between samples is temperature.
        np.random.seed(42)
        sample = generate_text(
            model, seed, GENERATE_LENGTH, temperature, char_to_id, id_to_char
        )
        print("\n" + "-" * 60)
        print(f"Temperature {temperature}")
        print("-" * 60)
        print(sample)
        lines.append(f"Temperature {temperature}")
        lines.append(sample)
        lines.append("")

    out_path = os.path.join("outputs", "q1_generated_text.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
