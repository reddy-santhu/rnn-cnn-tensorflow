# Home Assignment 2

**Student Name:** Santhosh Reddy Pasnoor  
**Student ID:** 700791582  
**Course:** CS5720 Neural Network and Deep Learning  
**Semester:** Fall 2026  
**University:** University of Central Missouri  

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 is what I used. TensorFlow does not run on 3.14.

```bash
python 01_text_generation.py
python 02_sentiment_classification.py
python 03_convolution.py
python 04_feature_extraction.py
python 05_cnn_architectures.py
```

Printouts and the Sobel figure go in `outputs/`. The first two scripts download their data the first time they run.

## Q1: Text Generation

`01_text_generation.py` loads Shakespeare and keeps the first 20,000 characters. Each training example is 40 characters, and the label is the character that comes next. An embedding turns those character ids into vectors, then an LSTM predicts the next id.

After training, the script starts from a short seed and adds one character at a time. The model outputs a score for every character. Temperature divides those scores before softmax.

At 1.0 the sample uses the probabilities the model learned. Below 1 the likely characters get even more of the probability, so the text repeats common letters. Above 1 the unlikely characters get more of a chance, so the text is messier. I saved samples at 0.5, 1.0, and 1.5 in `outputs/q1_generated_text.txt`.

## Q2: Sentiment Classification

`02_sentiment_classification.py` loads the IMDB reviews, keeps the 10,000 most common words, and pads every review to length 150. The model is an embedding, an LSTM, and one sigmoid output for positive or negative.

It prints a confusion matrix and a classification report. On this run the test accuracy was 0.7950. For the negative class, precision was 0.7467 and recall was 0.9207. For the positive class, precision was 0.8829 and recall was 0.6569.

Precision is the share of reviews the model calls positive that really are positive. Recall is the share of truly positive reviews it finds. Raising the cutoff makes fewer positive predictions, so precision goes up and recall goes down. Lowering it does the opposite. Accuracy stayed near 0.80 here while the positive class was still missing a lot of reviews, which is why the report matters more than accuracy alone.

The full report is in `outputs/q2_classification_report.txt`.

## Q3: Convolution

`03_convolution.py` uses the 5x5 matrix and 3x3 kernel from the sheet. It runs `tf.nn.conv2d` four times: stride 1 and stride 2, each with VALID and SAME padding.

The VALID maps come out all zeros. The kernel is a Laplacian, and the input goes up by 1 on every step, so those differences cancel inside the matrix. SAME padding fills the outside with zeros, so the border values are not zero. The four maps are in `outputs/q3_feature_maps.txt`.

## Q4: Filters and Pooling

`04_feature_extraction.py` builds a grayscale image and applies the Sobel X and Sobel Y kernels from the sheet with OpenCV. Sobel X picks up vertical edges. Sobel Y picks up horizontal edges. The original and both filtered images are in `outputs/q4_sobel_edges.png`.

The same script makes a random 4x4 matrix and runs 2x2 max pooling and 2x2 average pooling. Max pooling keeps the largest value in each window. Average pooling keeps the mean. All three matrices are printed in `outputs/q4_pooling.txt`.

## Q5: AlexNet and ResNet

`05_cnn_architectures.py` builds the AlexNet layers from the sheet and prints `model.summary()`. The input is 227x227x3 so the valid-padding layers still have a positive size. The last layer is a 10-way softmax. The summary has 24,767,882 parameters.

`residual_block` takes an input and a filter count, applies two 3x3 convolutions, and adds the input to that output before the ReLU. The ResNet model starts with a 7x7 convolution, stride 2, 64 filters, then two of those blocks, a flatten, a dense layer of 128, and a softmax output. That summary has 8,547,210 parameters.

- `outputs/q5_alexnet_summary.txt`
- `outputs/q5_resnet_summary.txt`
