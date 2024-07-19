#include "mnist.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t read_be_i32(FILE* file) {
    uint32_t num;
    fread(&num, sizeof num, 1, file);
#ifndef BIG_ENDIAN
    num = __builtin_bswap32(num);
#endif
    return num;
}

int mnist_parse_images(const char* path, float* images, int64_t num_images) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        perror("fopen");
        return 1;
    }

    uint32_t magic = read_be_i32(file);
    if (magic != MNIST_IMAGE_MAGIC) {
        printf("invalid magic number: %u, expected %i for images\n", magic, MNIST_IMAGE_MAGIC);
        fclose(file);
        return 1;
    }

    uint32_t dims[3];
    for (int i = 0; i < 3; i++) {
        dims[i] = read_be_i32(file);
    }

    if (dims[0] < num_images) {
        printf("not enough images; found %u, expected %lld\n", dims[0], num_images);
        fclose(file);
        return 1;
    }

    if (dims[1] != MNIST_IMAGE_DIM || dims[2] != MNIST_IMAGE_DIM) {
        printf("invalid image dimensions: %ux%u\n", dims[0], dims[1]);
        fclose(file);
        return 1;
    }

    int64_t images_size = dims[1] * dims[2] * num_images;
    uint8_t* images_u8 = malloc(images_size);
    if (!images_u8) {
        perror("malloc");
        fclose(file);
        return 1;
    }
    uint8_t* images_u8_end = images_u8 + fread(images_u8, 1, images_size, file) + 1;
    fclose(file);

    for (; images_u8 != images_u8_end; images_u8++, images++) {
        *images = *images_u8 / 255.0f;
    }

    free(images_u8);
    return 0;
}

int mnist_parse_labels(const char* path, int8_t* labels, int64_t num_labels) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        perror("fopen");
        return 1;
    }

    uint32_t magic = read_be_i32(file);
    if (magic != MNIST_LABEL_MAGIC) {
        printf("invalid magic number: %u, expected %i for labels\n", magic, MNIST_LABEL_MAGIC);
        fclose(file);
        return 1;
    }

    uint32_t num_labels_file = read_be_i32(file);
    if (num_labels_file < num_labels) {
        printf("not enough labels; found %u, expected %lld\n", num_labels_file, num_labels);
        fclose(file);
        return 1;
    }

    fread(labels, 1, num_labels, file);
    fclose(file);

    return 0;
}
