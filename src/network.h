#pragma once

#include <stdint.h>

typedef struct network {
    float* weights;
    float* biases;
    int64_t num_weights;
    int64_t num_biases;

    int64_t num_layers;
    int64_t* layer_sizes;

    int64_t* weight_mat_offsets;
    int64_t* bias_vec_offsets;
} network;

void nn_init(network* nn, int64_t num_layers, int64_t* layer_sizes);
void nn_free(network* nn);

void nn_train(network* nn, const float* train_data, const int8_t* labels, int64_t num_samples,
              int64_t num_epochs, int64_t mini_batch_size, float eta);
int64_t nn_evaluate(network* nn, const float* test_data, const int8_t* labels, int64_t num_samples);
void nn_feedforward(network* nn, float* output, float* z_output, const float* input,
                    int64_t num_inputs);

int nn_save(network* nn, const char* path);
int nn_load(network* nn, const char* path);
