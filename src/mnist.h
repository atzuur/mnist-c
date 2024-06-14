#pragma once

#include <limits.h>
#include <stdint.h>

_Static_assert(CHAR_BIT == 8, ":sob:");

#if defined __BYTE_ORDER__
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define BIG_ENDIAN
#endif
#else
#error "please define the macro __BYTE_ORDER__"
#endif

#define MNIST_IMAGE_DIM 28
#define MNIST_IMAGE_MAGIC 0x803
#define MNIST_LABEL_MAGIC 0x801

int parse_mnist_images(const char* path, float* images, int64_t num_images);
int parse_mnist_labels(const char* path, int8_t* labels, int64_t num_labels);
