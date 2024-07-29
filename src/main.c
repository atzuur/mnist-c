#include "mnist.h"
#include "network.h"
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    const int64_t NUM_TRAIN_SAMPLES = 60000;
    const int64_t NUM_TEST_SAMPLES = 10000;
    const int64_t SAMPLE_LEN = MNIST_IMAGE_DIM * MNIST_IMAGE_DIM;

    float* train_data = malloc(NUM_TRAIN_SAMPLES * SAMPLE_LEN * sizeof *train_data);
    int ret =
        mnist_parse_images("train-data/train-images.idx3-ubyte", train_data, NUM_TRAIN_SAMPLES);
    if (ret) {
        return ret;
    }

    int8_t* train_labels = malloc(NUM_TRAIN_SAMPLES * sizeof *train_labels);
    ret = mnist_parse_labels("train-data/train-labels.idx1-ubyte", train_labels, NUM_TRAIN_SAMPLES);
    if (ret) {
        return ret;
    }

    float* test_data = malloc(NUM_TEST_SAMPLES * SAMPLE_LEN * sizeof *test_data);
    ret = mnist_parse_images("test-data/t10k-images.idx3-ubyte", test_data, NUM_TEST_SAMPLES);
    if (ret) {
        return ret;
    }

    int8_t* test_labels = malloc(NUM_TEST_SAMPLES * sizeof *test_labels);
    ret = mnist_parse_labels("test-data/t10k-labels.idx1-ubyte", test_labels, NUM_TEST_SAMPLES);
    if (ret) {
        return ret;
    }

    network nn;
    int64_t layer_sizes[] = {SAMPLE_LEN, 30, 10};
    nn_init(&nn, 3, layer_sizes);
    nn_train(&nn, train_data, train_labels, NUM_TRAIN_SAMPLES, 30, 10, 3.0f);

    int64_t correct = nn_evaluate(&nn, test_data, test_labels, NUM_TEST_SAMPLES);
    double acc = (double)correct / NUM_TEST_SAMPLES * 100.0;
    printf("accuracy: %" PRId64 "/%" PRId64 " (%.2lf%%)\n", correct, NUM_TEST_SAMPLES, acc);

    nn_free(&nn);

    free(train_data);
    free(train_labels);
    free(test_data);
    free(test_labels);
}
