#include "mnist.h"
#include "network.h"
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int64_t MNIST_SAMPLE_LEN = MNIST_IMAGE_DIM * MNIST_IMAGE_DIM;
const int64_t NUM_TRAIN_SAMPLES = 60000;
const int64_t NUM_TEST_SAMPLES = 10000;

static int train_mnist(network* nn, int64_t epochs, int64_t mini_batch_size, float eta) {
    float* train_data = malloc(NUM_TRAIN_SAMPLES * MNIST_SAMPLE_LEN * sizeof *train_data);
    float* test_data = malloc(NUM_TEST_SAMPLES * MNIST_SAMPLE_LEN * sizeof *test_data);
    int8_t* train_labels = malloc(NUM_TRAIN_SAMPLES * sizeof *train_labels);
    int8_t* test_labels = malloc(NUM_TEST_SAMPLES * sizeof *test_labels);

    int ret;
    ret = mnist_parse_images("data/train-images.idx3-ubyte", train_data, NUM_TRAIN_SAMPLES);
    ret = mnist_parse_labels("data/train-labels.idx1-ubyte", train_labels, NUM_TRAIN_SAMPLES);
    ret = mnist_parse_images("data/t10k-images.idx3-ubyte", test_data, NUM_TEST_SAMPLES);
    ret = mnist_parse_labels("data/t10k-labels.idx1-ubyte", test_labels, NUM_TEST_SAMPLES);
    if (ret) {
        goto end;
    }

    nn_train(nn, train_data, train_labels, NUM_TRAIN_SAMPLES, epochs, mini_batch_size, eta);
    int64_t correct = nn_evaluate(nn, test_data, test_labels, NUM_TEST_SAMPLES);
    double acc = (double)correct / NUM_TEST_SAMPLES * 100.0;
    printf("accuracy: %" PRId64 "/%" PRId64 " (%.2lf %%)\n", correct, NUM_TEST_SAMPLES, acc);

    ret = nn_save(nn, "mnist.model");
end:
    free(train_data);
    free(train_labels);
    free(test_data);
    free(test_labels);

    return ret;
}

static int test_mnist(network* nn, const char* image) {
    int ret = nn_load(nn, "mnist.model");
    if (ret) {
        return ret;
    }
    FILE* image_file;
    if (strcmp(image, "-") == 0) {
        image_file = stdin;
    } else {
        image_file = fopen(image, "rb");
        if (!image_file) {
            perror("fopen");
            return 1;
        }
    }

    float* image_data = malloc(nn->layer_sizes[0] * sizeof *image_data);
    if (fread(image_data, sizeof *image_data, MNIST_SAMPLE_LEN, image_file) !=
        (size_t)MNIST_SAMPLE_LEN) {
        free(image_data);
        perror("fread");
        return 1;
    }

    int64_t output_len = nn->layer_sizes[nn->num_layers - 1];
    float* output = malloc(output_len * sizeof *output);
    nn_feedforward(nn, output, NULL, image_data, 1);

    printf("prediction results:\n");
    for (int64_t i = 0; i < output_len; i++) {
        printf("%" PRId64 ": ", i);
        int64_t n_chars = (int64_t)llroundf(output[i] * 20.0f) + 1;
        for (int64_t j = 0; j < n_chars; j++) {
            putchar('-');
        }
        printf(" %.1f %%\n", output[i] * 100);
    }

    free(image_data);
    free(output);
    return 0;
}

int main(int argc, const char** argv) {
    if (argc < 2) {
        printf("usage: %s train|<image>", argv[0]);
        return 1;
    }

    network nn;
    int64_t layer_sizes[] = {MNIST_SAMPLE_LEN, 50, 40, 10};
    nn_init(&nn, 4, layer_sizes);

    int ret;
    if (strcmp(argv[1], "train") == 0) {
        ret = train_mnist(&nn, 30, 10, 3.0f);
    } else {
        ret = test_mnist(&nn, argv[1]);
    }

    nn_free(&nn);
    return ret;
}
