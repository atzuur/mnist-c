#include "network.h"
#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h> // clang :(
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static float sigmoid(float x) {
    return 1.0f / (expf(-x) + 1);
}

static float d_sigmoid(float x) {
    return sigmoid(x) * (1.0f - sigmoid(x));
}

static float d_quadr_cost(float a, int64_t y) {
    return a - (float)y;
}

static uint64_t x = 0;
static void rand_seed(uint64_t seed) {
    x = seed;
}

static uint64_t xorshift_star() {
    x ^= x >> 12; // a
    x ^= x << 25; // b
    x ^= x >> 27; // c
    return x * 2685821657736338717ULL;
}

static float rand_float() {
    return (float)xorshift_star() / (float)UINT64_MAX * 2.0f - 1.0f;
}

static int64_t rand_i64(int64_t max) {
    return xorshift_star() / (UINT64_MAX / (max + 1));
}

static float dot(const float* u, const float* v, int64_t len) {
    float prod = 0.0f;
    for (int64_t i = 0; i < len; i++) {
        prod += u[i] * v[i];
    }
    return prod;
}

static void get_z_matrix(network* nn, float* dest, int64_t layer, const float* inputs,
                         int64_t num_inputs) {
    assert(layer > 0 && layer < nn->num_layers);

    int64_t input_len = nn->layer_sizes[layer - 1];
    int64_t num_outputs = nn->layer_sizes[layer];
    const float* weights = nn->weights + nn->weight_mat_offsets[layer - 1];
    const float* biases = nn->biases + nn->bias_vec_offsets[layer - 1];

    for (int64_t i = 0; i < num_inputs; i++) {
        for (int64_t j = 0; j < num_outputs; j++) {
            const float* in_row = inputs + i * input_len;
            const float* w_row = weights + j * input_len;
            dest[i * num_outputs + j] = dot(in_row, w_row, input_len) + biases[j];
        }
    }
}

static void feedforward(network* nn, float* output, float* z_output, const float* input,
                        int64_t num_inputs) {
    // cast so `free(prev_layer)` is allowed, which is never done when `prev_layer == input`
    float* prev_layer = (float*)input;
    int64_t z_out_offset = 0;
    int64_t layer_size;
    for (int l = 1; l < nn->num_layers; l++) {
        layer_size = num_inputs * nn->layer_sizes[l];
        float* layer = malloc(layer_size * sizeof *layer);
        get_z_matrix(nn, layer, l, prev_layer, num_inputs);
        if (z_output) {
            memcpy(z_output + z_out_offset, layer, layer_size * sizeof *z_output);
            z_out_offset += layer_size;
        }
        for (int64_t i = 0; i < layer_size; i++) {
            layer[i] = sigmoid(layer[i]);
        }

        if (l != 1) {
            free(prev_layer);
        }
        prev_layer = layer;
    }

    memcpy(output, prev_layer, layer_size * sizeof *output);
    free(prev_layer);
}

static void swap(void* a, void* b, size_t size) {
    uint8_t* ap = a;
    uint8_t* bp = b;
    for (size_t i = 0; i < size; i++) {
        uint8_t tmp = ap[i];
        ap[i] = bp[i];
        bp[i] = tmp;
    }
}

static void* clone_buf(const void* buf, size_t size) {
    void* clone = malloc(size);
    return memcpy(clone, buf, size);
}

static void shuffle_train_data(float* data, int8_t* labels, int64_t num_samples,
                               int64_t sample_len) {
    for (int64_t i = 0; i < num_samples; i++) {
        int64_t new_idx = rand_i64(num_samples);
        swap(labels + i, labels + new_idx, sizeof *labels);
        swap(data + sample_len * i, data + sample_len * new_idx, sizeof *data * sample_len);
    }
}

static void transpose(float* dest, const float* src, int64_t num_rows, int64_t num_cols) {
    for (int64_t i = 0; i < num_rows; i++) {
        for (int64_t j = 0; j < num_cols; j++) {
            dest[j * num_rows + i] = src[i * num_cols + j];
        }
    }
}

void nn_init(network* nn, int64_t num_layers, const int64_t* layer_sizes) {
    assert(num_layers >= 2);

    nn->num_weights = 0;
    nn->num_biases = 0;
    nn->weight_mat_offsets = calloc(num_layers - 1, sizeof *nn->weight_mat_offsets);
    nn->bias_vec_offsets = calloc(num_layers - 1, sizeof *nn->bias_vec_offsets);
    for (int64_t i = 1; i < num_layers; i++) {
        nn->num_weights += layer_sizes[i] * layer_sizes[i - 1];
        nn->num_biases += layer_sizes[i];
        if (i < num_layers - 1) {
            nn->weight_mat_offsets[i] = nn->num_weights;
            nn->bias_vec_offsets[i] = nn->num_biases;
        }
    }

    nn->weights = malloc(nn->num_weights * sizeof *nn->weights);
    nn->biases = malloc(nn->num_biases * sizeof *nn->biases);

    for (int64_t i = 0; i < nn->num_weights; i++) {
        nn->weights[i] = rand_float();
    }
    for (int64_t i = 0; i < nn->num_biases; i++) {
        nn->biases[i] = rand_float();
    }

    nn->num_layers = num_layers;
    nn->layer_sizes = layer_sizes;
}

static void do_mini_batch(network* nn, const float* train_data, const int8_t* labels,
                          int64_t num_samples, float eta) {
    assert(num_samples > 0);

    int64_t total_neurons = nn->num_biases;
    int64_t last = nn->num_layers - 1;

    float* z_mats = malloc(total_neurons * num_samples * sizeof *z_mats);
    float* a_last = malloc(nn->layer_sizes[last] * num_samples * sizeof *a_last);
    feedforward(nn, a_last, z_mats, train_data, num_samples);

    int64_t first_weights_size = nn->layer_sizes[0] * nn->layer_sizes[1];
    float* bp_weights = malloc((nn->num_weights - first_weights_size) * sizeof *bp_weights);
    for (int64_t l = 2; l < nn->num_layers; l++) {
        transpose(bp_weights + nn->weight_mat_offsets[l - 1] - first_weights_size,
                  nn->weights + nn->weight_mat_offsets[l - 1], nn->layer_sizes[l],
                  nn->layer_sizes[l - 1]);
    }

    float* delta_mats = malloc(total_neurons * num_samples * sizeof *delta_mats);
    int64_t offset = total_neurons * num_samples;
    int64_t offset_prev;
    // backpropagation
    for (int64_t l = last; l > 0; l--) {
        offset_prev = offset;
        offset -= nn->layer_sizes[l] * num_samples;
        for (int64_t i = 0; i < num_samples; i++) {
            for (int64_t j = 0; j < nn->layer_sizes[l]; j++) {
                int64_t idx = i * nn->layer_sizes[l] + j;
                float del;
                if (l == last) {
                    del = d_quadr_cost(a_last[idx], labels[idx]) * d_sigmoid(z_mats[offset + idx]);
                } else {
                    float* w_mat_lp1 = bp_weights + nn->weight_mat_offsets[l] - first_weights_size;
                    float* jth_weight_col = w_mat_lp1 + j * nn->layer_sizes[l + 1];
                    float* ith_delta_row = delta_mats + offset_prev + i * nn->layer_sizes[l + 1];
                    del = dot(jth_weight_col, ith_delta_row, nn->layer_sizes[l + 1]) *
                          d_sigmoid(z_mats[offset + idx]);
                }
                delta_mats[offset + idx] = del;
            }
        }
    }

    assert(offset == 0);
    // gradient descent
    for (int64_t l = 1; l < nn->num_layers; l++) {
        for (int64_t j = 0; j < nn->layer_sizes[l]; j++) {
            for (int64_t k = 0; k < nn->layer_sizes[l - 1]; k++) {
                int64_t w_idx = nn->weight_mat_offsets[l - 1] + j * nn->layer_sizes[l] + k;
                float w_delta = 0.0f;
                for (int64_t i = 0; i < num_samples; i++) {
                    int64_t prev_idx = i * nn->layer_sizes[l - 1] + k;
                    float kth_a =
                        l == 1 ? train_data[prev_idx] : sigmoid(z_mats[offset_prev + prev_idx]);
                    float jth_delta = delta_mats[offset + i * nn->layer_sizes[l] + j];
                    w_delta += kth_a * jth_delta;
                }
                nn->weights[w_idx] -= eta / num_samples * w_delta;
            }
            float b_delta = 0.0f;
            for (int64_t i = 0; i < num_samples; i++) {
                b_delta += delta_mats[offset + i * nn->layer_sizes[l] + j];
            }
            nn->biases[nn->bias_vec_offsets[l - 1] + j] -= eta / num_samples * b_delta;
        }
        offset_prev = offset;
        offset += nn->layer_sizes[l] * num_samples;
    }

    free(z_mats);
    free(a_last);
    free(bp_weights);
    free(delta_mats);
}

void nn_train(network* nn, const float* train_data, const int8_t* labels, int64_t num_samples,
              int64_t num_epochs, int64_t mini_batch_size, float eta) {
    assert(num_samples % mini_batch_size == 0);

    int64_t sample_len = nn->layer_sizes[0];
    float* tdata = clone_buf(train_data, num_samples * sample_len * sizeof *tdata);
    int8_t* tlabels = clone_buf(labels, num_samples * sizeof *tlabels);

    int64_t num_outputs = nn->layer_sizes[nn->num_layers - 1];
    int8_t* label_vecs = calloc(num_samples * num_outputs, sizeof *label_vecs);

    for (int64_t e = 0; e < num_epochs; e++) {
        clock_t start = clock();
        shuffle_train_data(tdata, tlabels, num_samples, sample_len);
        for (int64_t i = 0; i < num_samples; i++) {
            label_vecs[i * num_outputs + tlabels[i]] = 1;
        }
        double shuffle_time = (double)(clock() - start) / CLOCKS_PER_SEC;
        for (int64_t m = 0; m < num_samples; m += mini_batch_size) {
            do_mini_batch(nn, tdata + m * sample_len, label_vecs + m * num_outputs, mini_batch_size,
                          eta);
        }
        double total_time = (double)(clock() - start) / CLOCKS_PER_SEC;
        printf("epoch %" PRId64 " took %.2lfs (of which %.2lfs shuffling)\n", e + 1, total_time,
               shuffle_time);
        fflush(stdout);
    }

    free(tdata);
    free(tlabels);
}

void nn_free(network* nn) {
    free(nn->weights);
    free(nn->biases);
    free(nn->weight_mat_offsets);
    free(nn->bias_vec_offsets);
}

static bool test_float(float a, float b) {
    if (fabsf(a - b) > FLT_EPSILON) {
        printf("mismatch: %f != %f\n", a, b);
        return false;
    }
    return true;
}

static void test_shuffle_train_data() {
    float inputs[] = {1.0f, 2.0f, 1.5f, 2.5f, 3.0f, 4.0f};
    int8_t labels[] = {1, 2, 3};
    rand_seed(69);
    float expected_inputs[] = {1.5f, 2.5f, 1.0f, 2.0f, 3.0f, 4.0f};
    int8_t expected_labels[] = {2, 1, 3};

    shuffle_train_data(inputs, labels, 3, 2);
    assert(memcmp(inputs, expected_inputs, 2 * 3 * sizeof *inputs) == 0);
    assert(memcmp(labels, expected_labels, 3 * sizeof *labels) == 0);
}

static void test_get_z_matrix() {
    network nn;
    nn_init(&nn, 2, (int64_t[]) {2, 2});

    memcpy(nn.weights, (float[]) {0.5f, 2.0f, 0.25f, 1.0f}, 2 * 2 * sizeof *nn.weights);
    memcpy(nn.biases, (float[]) {-0.25f, -1.0f}, 2 * sizeof *nn.biases);

    float inputs[] = {1.0f, 2.0f, 2.0f, 4.0f};
    float expected[] = {4.25f, 1.25f, 8.75f, 3.5f};
    float dest[4];
    get_z_matrix(&nn, dest, 1, inputs, 2);
    bool passed = true;
    for (int64_t i = 0; i < 4; i++) {
        passed = passed ? test_float(dest[i], expected[i]) : passed;
    }
    assert(passed);
    nn_free(&nn);
}

static void test_feedforward() {
    rand_seed(420);

    network nn;
    nn_init(&nn, 4, (int64_t[]) {8, 4, 4, 2});

    float inputs[2 * 8];
    for (int64_t i = 0; i < 2 * 8; i++) {
        inputs[i] = rand_float();
    }

    float expected[] = {0.25896616f, 0.362451f, 0.23493582f, 0.36595363f};
    float dest[4];
    feedforward(&nn, dest, NULL, inputs, 2);
    bool passed = true;
    for (int64_t i = 0; i < 4; i++) {
        passed = passed ? test_float(dest[i], expected[i]) : passed;
    }
    assert(passed);
    nn_free(&nn);
}

static void test_transpose() {
    float src[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dest[9];
    float expected[] = {1.0f, 5.0f, 2.0f, 6.0f, 3.0f, 7.0f, 4.0f, 8.0f};
    transpose(dest, src, 2, 4);
    bool passed = true;
    for (int64_t i = 0; i < 8; i++) {
        passed = passed ? dest[i] == expected[i] : passed;
    }
    assert(passed);
}

static void test_do_mini_batch() {
    network nn;
    nn_init(&nn, 3, (int64_t[]) {4, 4, 2});

    memcpy(nn.weights,
           (float[]) {1.0f,  2.0f,  3.0f,  4.0f,  5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                      13.0f, 14.0f, 15.0f, 16.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,  7.0f,  8.0f},
           (4 * 4 + 2 * 4) * sizeof *nn.weights);
    memcpy(nn.biases, (float[]) {-1.5f, -0.5f, 0.5f, 1.5f, -2.0f, -1.0f},
           (4 + 2) * sizeof *nn.biases);

    float data[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 1.5f, 2.5f, 3.5f, 4.5f,
    };
    int8_t labels[] = {0, 1, 1, 0};

    float expected_w[] = {1.0f,        2.0f,        3.0f,  4.0f,  5.0f,        6.0f,
                          7.0f,        8.0f,        9.0f,  10.0f, 11.0f,       12.0f,
                          13.0f,       14.0f,       15.0f, 16.0f, 0.99983249f, 1.99983249f,
                          2.99983249f, 3.99983249f, 5.0f,  6.0f,  7.0f,        8.0f};
    float expected_b[] = {-1.5f, -0.5f, 0.5f, 1.5f, -2.00016751f, -1.0f};
    do_mini_batch(&nn, data, labels, 2, 1.0f);

    bool passed = true;
    for (int64_t i = 0; i < 4 * 4 + 2 * 4; i++) {
        passed = passed ? test_float(nn.weights[i], expected_w[i]) : passed;
    }
    for (int64_t i = 0; i < 4 + 2; i++) {
        passed = passed ? test_float(nn.biases[i], expected_b[i]) : passed;
    }
    assert(passed);
    nn_free(&nn);
}

// int main() {
//     test_shuffle_train_data();
//     test_get_z_matrix();
//     test_feedforward();
//     test_transpose();
//     test_do_mini_batch();
// }
