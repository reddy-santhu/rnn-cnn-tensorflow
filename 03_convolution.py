"""
CS5720 Neural Network and Deep Learning - Home Assignment 2
Q3: Convolution with different stride and padding

Convolve a 5x5 input with a 3x3 kernel at stride 1 and 2,
using VALID and SAME padding.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

os.makedirs("outputs", exist_ok=True)

# Row-major values 1 through 25, as given in the assignment.
INPUT_MATRIX = np.array(
    [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ],
    dtype=np.float32,
)

# Discrete Laplacian. Keras / tf.nn.conv2d do not flip this kernel.
KERNEL = np.array(
    [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ],
    dtype=np.float32,
)

CASES = (
    (1, "VALID"),
    (1, "SAME"),
    (2, "VALID"),
    (2, "SAME"),
)


def convolve(stride, padding):
    """tf.nn.conv2d expects NHWC input and HWIO filters."""
    image = INPUT_MATRIX.reshape(1, 5, 5, 1)
    filters = KERNEL.reshape(3, 3, 1, 1)
    output = tf.nn.conv2d(
        image,
        filters,
        strides=[1, stride, stride, 1],
        padding=padding,
    )
    return output.numpy()[0, :, :, 0]


def format_map(feature_map):
    rows = []
    for row in feature_map:
        rows.append("  " + "  ".join(f"{value:7.1f}" for value in row))
    return "\n".join(rows)


def main():
    print("=" * 60)
    print("Q3: Convolution, stride, and padding")
    print("=" * 60)
    print("Input")
    print(format_map(INPUT_MATRIX))
    print("\nKernel")
    print(format_map(KERNEL))

    # VALID maps are all zeros: this kernel is a Laplacian, and the input
    # increases by 1 across each row, so interior second differences are 0.
    # SAME padding uses zeros outside the matrix, so the border is not zero.
    lines = ["Q3 convolution feature maps", ""]
    for stride, padding in CASES:
        feature_map = convolve(stride, padding)
        title = f"Stride = {stride}, Padding = {padding}"
        print("\n" + title)
        print(f"Output shape: {feature_map.shape}")
        print(format_map(feature_map))
        lines.append(title)
        lines.append(f"Output shape: {feature_map.shape}")
        lines.append(format_map(feature_map))
        lines.append("")

    out_path = os.path.join("outputs", "q3_feature_maps.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
