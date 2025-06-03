#include "../../include/settings.h"
#include "../../include/data_utils.h"
#include "../../include/rk4.cuh"

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <math.h>

// Apply drop at coordinates (x, y).
void applyDrop(float *mat, float *grid_x, float *grid_y, float x = 0.0, float y = 0.0, float k = -1.0)
{
    for (int i = 0; i < nx * ny; i++)
        mat[i] += k * perturb * exp(-(pow(grid_x[i] - x, 2) + pow(grid_y[i] - y, 2)));
}

// Apply step with slope angle k (degree) and distance d (from center (0, 0)).
void applyStep(float *mat, float *grid_x, float *grid_y, float k = 0.0, float d = 0.0)
{
    // Convert k to radian
    k = PI * k / 180.0;
    float sin_k = sin(k), cos_k = cos(k);
    for (int i = 0; i < nx * ny; i++)
    {
        if ((sin_k * grid_x[i] - cos_k * grid_y[i]) >= d)
            mat[i] += perturb;
    }
}

// Initialize fluid's surface and coriolis matrix.
void initilizer(float *mat, float *grid_x, float *grid_y, int argc, char **argv)
{
    int i;
    float x = 0.0, y = 0.0;

    // Set surface to base height
    for (i = 0; i < nx * ny; i++)
        mat[i] = H0;

    // No modification
    if (argc == 1)
    {
        printf("Initilize: CenterDrop (default)\n");
        applyDrop(mat, grid_x, grid_y);
        return;
    }

    // Process cmd input
    for (i = 1; i < argc; i++)
    {
        // Modifier: Drop, Args: float x, float y
        if (strcmp(argv[i], "-drop") == 0)
        {
            x = atof(argv[i + 1]), y = atof(argv[i + 2]);
            printf("Modify: Drop (x:%.2f, y:%.2f)\n", x, y);
            applyDrop(mat, grid_x, grid_y, x, y);
            i += 2;
        }
        // Modifier: Pinch, Args: float x, float y
        else if (strcmp(argv[i], "-pinch") == 0)
        {
            x = atof(argv[i + 1]), y = atof(argv[i + 2]);
            printf("Modify: Pinch (x:%.2f, y:%.2f)\n", x, y);
            applyDrop(mat, grid_x, grid_y, x, y, 1.0);
            i += 2;
        }
        // Modifier: Step, Args: float k, float d
        else if (strcmp(argv[i], "-step") == 0)
        {
            x = atof(argv[i + 1]), y = atof(argv[i + 2]);
            printf("Modify: Step (k:%.2f, d:%.2f)\n", x, y);
            applyStep(mat, grid_x, grid_y, x, y);
            i += 2;
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
    // Initialize with option from command line
    initilizer(eta, grid_x, grid_y, argc, argv);

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
