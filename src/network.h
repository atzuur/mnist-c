#pragma once

#include <stdint.h>

typedef struct network {
    float* weights;
    float* biases;
    int64_t num_weights;
    int64_t num_biases;

    int64_t num_layers;
    const int64_t* layer_sizes;

    int64_t* weight_mat_offsets;
    int64_t* bias_vec_offsets;
} network;

void nn_init(network* nn, int64_t num_layers, const int64_t* layer_sizes);
void nn_train(network* nn, const float* train_data, const int64_t* labels, int64_t num_samples,
              int64_t num_epochs, int64_t mini_batch_size, float eta);

void nn_free(network* nn);
