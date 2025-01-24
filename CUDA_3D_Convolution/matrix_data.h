#include "read_file.cpp"

#pragma once

#define TILE_WIDTH 16

void read3DInputFile(int &width, int &height, int &depth);

typedef struct 
{
    float *hostMemoryP;
    float *deviceMemoryP;
    int row;
    int col;
    int depth;
    size_t size;
} matrixData;


