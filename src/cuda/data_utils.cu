#include "../../include/data_utils.h"

#include <stdio.h>
#include <malloc.h>
#include <string.h>

void printMatrix(float *mat, int size_x, int size_y)
{
    for (int i = 0; i < size_y; i++)
    {
        for (int j = 0; j < size_x; j++)
            printf("%f ", *(mat + i * size_x + j));
        printf("\n");
    }
}

void printArray(float *arr, int size)
{
    for (int i = 0; i < size; i++)
        printf("%f ", *(arr + i));
    printf("\n");
}

// Fill an array with value.
void fillArray(float *arr, float value, int size)
{
    for (int i = 0; i < size; i++)
        arr[i] = value;
}

float *fullArray(float value, int size)
{
    float *arr = (float *)malloc(sizeof(float) * size);
    fillArray(arr, value, size);
    return arr;
}

float *lnspace(float start, float stop, int num)
{
    float step = (stop - start) / (num - 1);
    float *arr = (float *)malloc(sizeof(float) * num);
    for (int i = 0; i < num; i++)
        arr[i] = start + step * i;
    return arr;
}

void meshGrid2d(float *x, float *y, int size_x, int size_y, float **grid_x, float **grid_y)
{
    float *_grid_x = (float *)malloc(sizeof(float) * size_x * size_y);
    float *_grid_y = (float *)malloc(sizeof(float) * size_x * size_y);
    for (int i = 0; i < size_y; i++)
    {
        // Vertically stack x coordinates
        memcpy(_grid_x + i * size_x, x, sizeof(float) * size_x);
        // Horizontally stack y coordinates
        fillArray(_grid_y + i * size_x, y[i], size_x);
    }
    *grid_x = _grid_x, *grid_y = _grid_y;
}