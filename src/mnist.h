#pragma once

#include <limits.h>
#include <stdint.h>

#if defined __BYTE_ORDER__
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define MNIST_BIG_ENDIAN
#endif
#else
#error "please define the macro __BYTE_ORDER__"
#endif

#define MNIST_IMAGE_DIM 28
#define MNIST_IMAGE_MAGIC 0x803
#define MNIST_LABEL_MAGIC 0x801

int mnist_parse_images(const char* path, float* images, int64_t num_images);
int mnist_parse_labels(const char* path, int8_t* labels, int64_t num_labels);
