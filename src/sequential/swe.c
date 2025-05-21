#include "../../include/settings.h"
#include "../../include/data_utils.h"
#include "../../include/rk4.h"

#include <stdio.h>
#include <malloc.h>
#include <math.h>

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
    float *f = (float *)malloc(sizeof(float) * nx * ny);

    // Velocity in x direction.
    float *u = fullArray(0.0, nx * ny);

    // Velocity in y direction.
    float *v = fullArray(0.0, nx * ny);

    // Fluid depth.
    float *eta = (float *)malloc(sizeof(float) * nx * ny);

    // Set up coriolis matrix
    for (i = 0; i < nx; i++)
    {
        for (j = 0; j < ny; j++)
            *(f + i * nx + j) = 2.0 * omega * sin(PI * y[j] / Ly);
    }
    // Pertubation
    for (i = 0; i < nx * ny; i++)
    {
        eta[i] = H0 - perturb * exp(-(pow(grid_x[i], 2) + pow(grid_y[i], 2)));
    }
    free(x), free(y), free(grid_x), free(grid_y); // Free grid memory

    // Generate render data
    printf("Generating render data: 0/%d\r", nt - 1);
    writeRenderData(t[0], eta, nx * ny);
    j = 1; // Rendered count
    for (i = 1; i < nt; i++)
    {
        rk4(u, v, eta, f);
        if (i % renderStep == 0)
        {
            j += 1;
            printf("Generating render data: %d/%d, timestep = %f\r", j, renderCount, t[i]);
            writeRenderData(t[i], eta, nx * ny);
        }
    }
    // Freeing memory
    free(u), free(v), free(eta), free(f);
    return 0;
}
