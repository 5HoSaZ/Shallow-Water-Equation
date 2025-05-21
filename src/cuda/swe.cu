#include "../../include/settings.h"
#include "../../include/data_utils.h"
#include "../../include/rk4.cuh"

#include <stdio.h>
#include <malloc.h>
#include <math.h>

#define IniCenterDrop 0
#define IniStepX 1
#define IniPinchDrop 2

// Initialize fluid's surface and coriolis matrix.
void initilizer(float *mat, float *grid_x, float *grid_y, int kind)
{
    int i;
    if (kind == IniCenterDrop)
    {

        for (i = 0; i < nx * ny; i++)
            mat[i] = H0 - perturb * exp(-(pow(grid_x[i], 2) + pow(grid_y[i], 2)));
    }
    else if (kind == IniStepX)
    {
        for (i = 0; i < nx * ny; i++)
        {
            if (grid_x[i] < 0)
                mat[i] = H0 + perturb;
            else
                mat[i] = H0;
        }
    }
    else if (kind == IniPinchDrop)
    {
        for (i = 0; i < nx * ny; i++)
        {
            mat[i] = H0;
            mat[i] -= perturb * exp(-(pow(grid_x[i] - 2, 2) + pow(grid_y[i] - 2, 2)));
            mat[i] += perturb * exp(-(pow(grid_x[i] + 2, 2) + pow(grid_y[i] + 2, 2)));
        }
    }
}

// =============================================================================

// Write SWE render data to data/render/t=timestep.
void writeRenderData(float timestep, float *data, int size)
{
    char name[100];
    sprintf(name, "%s/t=%f", renderTmp, timestep);
    FILE *fptr = fopen(name, "wb");
    fwrite(data, sizeof(float) * size, 1, fptr);
    fclose(fptr);
}
// =============================================================================

int main(int argc, char *argv[])
{
    int i, j;

    // Model variables
    float *x = lnspace(x_start, x_end, nx); // x-coordinates.
    float *y = lnspace(y_start, y_end, ny); // y-coordinates.
    float *t = lnspace(t_start, t_end, nt); // Time steps.

    // 2D grid space.
    float *grid_x, *grid_y;
    meshGrid2d(x, y, nx, ny, &grid_x, &grid_y);

    // Coriolis matrix.
    float *f, *f_gpu;
    f = (float *)malloc(sizeof(float) * nx * ny);
    // Set up coriolis matrix
    for (i = 0; i < nx; i++)
    {
        for (j = 0; j < ny; j++)
            f[i * nx + j] = 2.0 * omega * sin(PI * y[j] / Ly);
    }
    cudaMalloc((void **)&f_gpu, sizeof(float) * nx * ny);
    cudaMemcpy(f_gpu, f, sizeof(float) * nx * ny, cudaMemcpyHostToDevice);
    free(f); // gpu only

    // Velocity in x direction.
    float *u, *u_gpu;
    u = fullArray(0.0, nx * ny);
    cudaMalloc((void **)&u_gpu, sizeof(float) * nx * ny);
    cudaMemcpy(u_gpu, u, sizeof(float) * nx * ny, cudaMemcpyHostToDevice);
    free(u); // gpu only

    // Velocity in y direction.
    float *v, *v_gpu;
    v = fullArray(0.0, nx * ny);
    cudaMalloc((void **)&v_gpu, sizeof(float) * nx * ny);
    cudaMemcpy(v_gpu, v, sizeof(float) * nx * ny, cudaMemcpyHostToDevice);
    free(v); // gpu only

    // Fluid depth.
    float *eta, *eta_gpu;
    eta = (float *)malloc(sizeof(float) * nx * ny);
    initilizer(eta, grid_x, grid_y, IniCenterDrop);
    // initilizer(eta, grid_x, grid_y, IniPinchDrop);
    cudaMalloc((void **)&eta_gpu, sizeof(float) * nx * ny);
    cudaMemcpy(eta_gpu, eta, sizeof(float) * nx * ny, cudaMemcpyHostToDevice);
    free(x), free(y), free(grid_x), free(grid_y); // Free grid memory

    // Generate render data
    printf("Generating render data: 0/%d\r", nt - 1);
    writeRenderData(t[0], eta, nx * ny);
    j = 1; // Rendered count
    for (i = 1; i < nt; i++)
    {
        rk4(u_gpu, v_gpu, eta_gpu, f_gpu);
        if (i % renderStep == 0)
        {
            j += 1;
            printf("Generating render data: %d/%d, timestep = %f\r", j, renderCount, t[i]);
            cudaMemcpy(eta, eta_gpu, sizeof(float) * nx * ny, cudaMemcpyDeviceToHost);
            writeRenderData(t[i], eta, nx * ny);
        }
    }
    // Freeing memory
    cudaFree(u_gpu), cudaFree(v_gpu), cudaFree(f_gpu);
    free(eta), cudaFree(eta_gpu);
    return 0;
}
