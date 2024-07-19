#include "mnist.h"
#include "network.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    const int NUM_TRAIN_SAMPLES = 60000;

    float* train_data =
        malloc(NUM_TRAIN_SAMPLES * MNIST_IMAGE_DIM * MNIST_IMAGE_DIM * sizeof *train_data);
    int ret =
        mnist_parse_images("train-data/train-images.idx3-ubyte", train_data, NUM_TRAIN_SAMPLES);
    if (ret) {
        return ret;
    }

    int8_t* train_labels = malloc(NUM_TRAIN_SAMPLES * sizeof *train_labels);
    mnist_parse_labels("train-data/train-labels.idx1-ubyte", train_labels, NUM_TRAIN_SAMPLES);
    if (ret) {
        return ret;
    }

    network nn;
    int64_t layer_sizes[] = {MNIST_IMAGE_DIM * MNIST_IMAGE_DIM, 30, 10};
    nn_init(&nn, 3, layer_sizes);
    nn_train(&nn, train_data, train_labels, NUM_TRAIN_SAMPLES, 30, 10, 3.0f);
}