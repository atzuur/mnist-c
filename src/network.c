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
    assert(max > 0);
    uint64_t mask = max;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;

    int64_t value;
    while ((value = xorshift_star() & mask) > max)
        ;
    return value;
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

void nn_feedforward(network* nn, float* output, float* z_output, const float* input,
                    int64_t num_inputs) {
    // cast so `free(prev_layer)` is allowed, which is never done when `prev_layer == input`
    float* prev_layer = (float*)input;
    int64_t z_out_offset = 0;
    int64_t layer_size = 0;
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
    for (int64_t i = num_samples - 1; i > 0; i--) {
        int64_t j = rand_i64(i);
        swap(labels + i, labels + j, sizeof *labels);
        swap(data + sample_len * i, data + sample_len * j, sizeof *data * sample_len);
    }
}

static void transpose(float* dest, const float* src, int64_t num_rows, int64_t num_cols) {
    for (int64_t i = 0; i < num_rows; i++) {
        for (int64_t j = 0; j < num_cols; j++) {
            dest[j * num_rows + i] = src[i * num_cols + j];
        }
    }
}

static int64_t argmax(const float* input, int64_t len) {
    assert(len >= 1);
    int64_t max = 0;
    for (int64_t i = 0; i < len; i++) {
        if (input[i] > input[max]) {
            max = i;
        }
    }
    return max;
}

void nn_init(network* nn, int64_t num_layers, int64_t* layer_sizes) {
    assert(num_layers >= 2);
    rand_seed(time(NULL));

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
    nn_feedforward(nn, a_last, z_mats, train_data, num_samples);

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
                    del = (a_last[idx] - labels[idx]) * d_sigmoid(z_mats[offset + idx]);
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
                int64_t w_idx = nn->weight_mat_offsets[l - 1] + j * nn->layer_sizes[l - 1] + k;
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

    clock_t start = clock();
    for (int64_t e = 1; e < num_epochs + 1; e++) {
        shuffle_train_data(tdata, tlabels, num_samples, sample_len);
        memset(label_vecs, 0, num_samples * num_outputs * sizeof *label_vecs);
        for (int64_t i = 0; i < num_samples; i++) {
            label_vecs[i * num_outputs + tlabels[i]] = 1;
        }
        for (int64_t m = 0; m < num_samples; m += mini_batch_size) {
            do_mini_batch(nn, tdata + m * sample_len, label_vecs + m * num_outputs, mini_batch_size,
                          eta);
        }
        double total_time = (double)(clock() - start) / CLOCKS_PER_SEC;
        printf("epoch %" PRId64 ", avg. epoch time: %.2lfs\r", e, total_time / e);
        fflush(stdout);
    }
    puts("");

    free(tdata);
    free(tlabels);
    free(label_vecs);
}

int64_t nn_evaluate(network* nn, const float* test_data, const int8_t* labels,
                    int64_t num_samples) {
    int64_t output_len = nn->layer_sizes[nn->num_layers - 1];
    float* output = malloc(output_len * num_samples * sizeof *output);
    nn_feedforward(nn, output, NULL, test_data, num_samples);

    int64_t correct = 0;
    for (int64_t i = 0; i < num_samples; i++) {
        int64_t pred = argmax(output + i * output_len, output_len);
        correct += pred == labels[i];
    }
    free(output);
    return correct;
}

int nn_save(network* nn, const char* path) {
    FILE* file = fopen(path, "wb");
    if (!file) {
        perror("fopen");
        return 1;
    }

    if (fwrite(&nn->num_layers, sizeof(int64_t), 1, file) != 1 ||
        fwrite(nn->layer_sizes, sizeof(int64_t), nn->num_layers, file) != (size_t)nn->num_layers ||
        fwrite(nn->weights, sizeof(float), nn->num_weights, file) != (size_t)nn->num_weights ||
        fwrite(nn->biases, sizeof(float), nn->num_biases, file) != (size_t)nn->num_biases) {
        perror("fwrite");
        return 1;
    }

    fclose(file);
    return 0;
}

int nn_load(network* nn, const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        perror("fopen");
        return 1;
    }

    if (fread(&nn->num_layers, sizeof(int64_t), 1, file) != 1 ||
        fread(nn->layer_sizes, sizeof(int64_t), nn->num_layers, file) != (size_t)nn->num_layers ||
        fread(nn->weights, sizeof(float), nn->num_weights, file) != (size_t)nn->num_weights ||
        fread(nn->biases, sizeof(float), nn->num_biases, file) != (size_t)nn->num_biases) {
        perror("fread");
        return 1;
    }

    fclose(file);
    return 0;
}

void nn_free(network* nn) {
    free(nn->weights);
    free(nn->biases);
    free(nn->weight_mat_offsets);
    free(nn->bias_vec_offsets);
}
