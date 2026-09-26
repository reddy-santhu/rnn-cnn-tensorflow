"""
CS5720 Neural Network and Deep Learning - Home Assignment 2
Q4: CNN feature extraction with filters and pooling

Task 1 applies the given Sobel kernels with OpenCV.
Task 2 applies 2x2 max pooling and average pooling with Keras.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

os.makedirs("outputs", exist_ok=True)

# Kernels from the assignment sheet.
SOBEL_X = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=np.float32,
)
SOBEL_Y = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ],
    dtype=np.float32,
)


def make_sample_image():
    """A small grayscale picture with a rectangle, a circle, and a diagonal."""
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(image, (30, 40), (140, 170), 220, thickness=-1)
    cv2.circle(image, (190, 90), 45, 160, thickness=-1)
    cv2.line(image, (20, 230), (240, 180), 255, thickness=4)
    return image


def save_edge_figure(original, sobel_x, sobel_y):
    """Write the three views the assignment asks to display."""
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = (
        (original, "Original Image", "gray"),
        (sobel_x, "Edge detection using Sobel-X", "gray"),
        (sobel_y, "Edge detection using Sobel-Y", "gray"),
    )
    for axis, (image, title, cmap) in zip(axes, panels):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    path = os.path.join("outputs", "q4_sobel_edges.png")
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def run_edge_detection():
    print("=" * 60)
    print("Task 1: Sobel edge detection")
    print("=" * 60)

    original = make_sample_image()
    # filter2D applies the exact kernels from the sheet. CV_32F keeps negative edges.
    response_x = cv2.filter2D(original, cv2.CV_32F, SOBEL_X)
    response_y = cv2.filter2D(original, cv2.CV_32F, SOBEL_Y)

    # Shift and scale only for display. The filter responses themselves can be negative.
    show_x = cv2.convertScaleAbs(response_x)
    show_y = cv2.convertScaleAbs(response_y)
    path = save_edge_figure(original, show_x, show_y)
    print(f"Saved {path}")
    return path


def format_matrix(matrix):
    rows = []
    for row in np.asarray(matrix):
        rows.append("  " + "  ".join(f"{value:7.2f}" for value in row))
    return "\n".join(rows)


def run_pooling():
    print("\n" + "=" * 60)
    print("Task 2: Max pooling and average pooling")
    print("=" * 60)

    # Integers from 0 to 19 are easier to check by hand than floats.
    values = tf.random.uniform(shape=(4, 4), minval=0, maxval=20, dtype=tf.int32)
    values = tf.cast(values, tf.float32)
    # Keras pooling layers expect a batch and a channel: (1, 4, 4, 1).
    image = values[tf.newaxis, :, :, tf.newaxis]

    max_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    avg_layer = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)
    max_pooled = max_layer(image).numpy()[0, :, :, 0]
    avg_pooled = avg_layer(image).numpy()[0, :, :, 0]
    original = values.numpy()

    print("Original 4x4")
    print(format_matrix(original))
    print("\nMax pooling 2x2")
    print(format_matrix(max_pooled))
    print("\nAverage pooling 2x2")
    print(format_matrix(avg_pooled))

    lines = [
        "Q4 pooling",
        "Original 4x4",
        format_matrix(original),
        "",
        "Max pooling 2x2",
        format_matrix(max_pooled),
        "",
        "Average pooling 2x2",
        format_matrix(avg_pooled),
        "",
    ]
    out_path = os.path.join("outputs", "q4_pooling.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"\nSaved {out_path}")
    return out_path


def main():
    run_edge_detection()
    run_pooling()


if __name__ == "__main__":
    main()
