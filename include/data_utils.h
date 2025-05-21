#ifndef _DATA_UTILS_H
#define _DATA_UTILS_H

// Print a matrix.
void printMatrix(float *mat, int size_x, int size_y);

// Print an array.
void printArray(float *arr, int size);

// Wrap an interger given left and right border.
int wrapi(int value, int lb, int rb);

// Return a slice of a matrix.
float *matrixSlice(float *mat, int size_x, int size_y, int x1, int x2, int y1, int y2);

// Set a matrix value by slice.
void matrixSet(float *mat, float *values, int size_x, int size_y, int x1, int x2, int y1, int y2);

// Fill an array with value.
void fillArray(float *arr, float value, int size);

// Return an array filled with value.
float *fullArray(float value, int size);

// Return an array of evenly spaced numbers over an interval.
float *lnspace(float start, float stop, int num);

// Generate a mesh of 2d coordinates from coordinate vectors.
void meshGrid2d(float *x, float *y, int size_x, int size_y, float **grid_x, float **grid_y);

#endif