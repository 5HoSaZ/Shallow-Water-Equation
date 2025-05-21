#include "../../include/rk4.cuh"

#include "../../include/settings.h"
#include "../../include/data_utils.h"

#include <stdio.h>
#include <malloc.h>

#define dt2 dt / 2
#define dt6 dt / 6
#define dx2 dx / 2
#define dy2 dy / 2

#define gdx2 g *dx2 // Gravity multiplier constants for gradient over x.
#define gdy2 g *dy2 // Gravity multiplier constants for gradient over y.

#define visdxx vis / (dx * dx) // Diffusion multiplier constants for diffusion over x.
#define visdyy vis / (dy * dy) // Diffusion multiplier constants for diffusion over y.

// Wrap index given range, return a triple of b-egin, c-enter and en-d.
__device__ void wrapIndex(int idx, int range, int *b, int *c, int *d)
{
    if (idx == 0)
        *b = range - 1, *c = idx, *d = idx + 1;
    else if (idx == range - 1)
        *b = idx - 1, *c = idx, *d = 0;
    else
        *b = idx - 1, *c = idx, *d = idx + 1;
}

// Get array value by 2d-index (program specific).
__device__ float getItem(float *arr, int x, int y)
{
    return *(arr + x + y * nx);
}

void rk4(float *u, float *v, float *eta, float *f)
{
    // Slopes
    float *su, *sv, *sh;
    cudaMalloc((void **)&su, sizeof(float) * nx * ny);
    cudaMalloc((void **)&sv, sizeof(float) * nx * ny);
    cudaMalloc((void **)&sh, sizeof(float) * nx * ny);

    // RK4 step 1 / 4
    float *ku1, *kv1, *kh1;
    cudaMalloc((void **)&ku1, sizeof(float) * nx * ny);
    cudaMalloc((void **)&kv1, sizeof(float) * nx * ny);
    cudaMalloc((void **)&kh1, sizeof(float) * nx * ny);
    rk4Solve<<<GridSize, BlockSize>>>(u, v, eta, f, ku1, kv1, kh1);

    // RK4 step 2 / 4
    float *ku2, *kv2, *kh2;
    cudaMalloc((void **)&ku2, sizeof(float) * nx * ny);
    cudaMalloc((void **)&kv2, sizeof(float) * nx * ny);
    cudaMalloc((void **)&kh2, sizeof(float) * nx * ny);
    rk4Slope<<<GridSize, BlockSize>>>(su, sv, sh, u, v, eta, ku1, kv1, kh1, dt2);
    rk4Solve<<<GridSize, BlockSize>>>(su, sv, sh, f, ku2, kv2, kh2);

    // RK4 step 3 / 4
    float *ku3, *kv3, *kh3;
    cudaMalloc((void **)&ku3, sizeof(float) * nx * ny);
    cudaMalloc((void **)&kv3, sizeof(float) * nx * ny);
    cudaMalloc((void **)&kh3, sizeof(float) * nx * ny);
    rk4Slope<<<GridSize, BlockSize>>>(su, sv, sh, u, v, eta, ku2, kv2, kh2, dt2);
    rk4Solve<<<GridSize, BlockSize>>>(su, sv, sh, f, ku3, kv3, kh3);

    // RK4 step 4 / 4
    float *ku4, *kv4, *kh4;
    cudaMalloc((void **)&ku4, sizeof(float) * nx * ny);
    cudaMalloc((void **)&kv4, sizeof(float) * nx * ny);
    cudaMalloc((void **)&kh4, sizeof(float) * nx * ny);
    rk4Slope<<<GridSize, BlockSize>>>(su, sv, sh, u, v, eta, ku3, kv3, kh3, dt);
    rk4Solve<<<GridSize, BlockSize>>>(su, sv, sh, f, ku4, kv4, kh4);

    // Free slope memory
    cudaFree(su), cudaFree(sv), cudaFree(sh);

    // Update variables
    rk4Update<<<GridSize, BlockSize>>>(u, ku1, ku2, ku3, ku4);
    rk4Update<<<GridSize, BlockSize>>>(v, kv1, kv2, kv3, kv4);
    rk4Update<<<GridSize, BlockSize>>>(eta, kh1, kh2, kh3, kh4);

    // Free step memory
    cudaFree(ku1), cudaFree(ku2), cudaFree(ku3), cudaFree(ku4);
    cudaFree(kv1), cudaFree(kv2), cudaFree(kv3), cudaFree(kv4);
    cudaFree(kh1), cudaFree(kh2), cudaFree(kh3), cudaFree(kh4);
}

__global__ void rk4Slope(float *su, float *sv, float *sh, float *u, float *v, float *h, float *ku, float *kv, float *kh, float d)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < nx * ny; i += stride)
    {
        su[i] = u[i] + ku[i] * d;
        sv[i] = v[i] + kv[i] * d;
        sh[i] = h[i] + kh[i] * d;
    }
}

__global__ void rk4Update(float *x, float *k1, float *k2, float *k3, float *k4)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < nx * ny; i += stride)
        x[i] += (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * dt6;
}

__global__ void rk4Solve(float *u_in, float *v_in, float *h_in, float *f, float *u_out, float *v_out, float *h_out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Iterating index.
    int i, ix, iy;

    // Begin, Center, enD.
    int xb, xc, xd, yb, yc, yd;

    // Advection terms in continuity and momentum equations.
    float hx, hy, ux, uy, vx, vy;

    // Coriolis terms in momentum equations.
    float fu, fv;

    // Gravity/Pressure gradient terms in momentum equations.
    float gx, gy;

    // Viscous drag terms in momentum equations.
    float ku, kv;

    // Diffusion terms in momentum equations.
    float uxx, uyy, vxx, vyy;

    // Calculation loop
    for (i = index; i < nx * ny; i += stride)
    {
        ix = i % nx, iy = i / nx;
        wrapIndex(ix, nx, &xb, &xc, &xd);
        wrapIndex(iy, ny, &yb, &yc, &yd);

        // hx, hy term in continuity equation
        hx = (getItem(h_in, xd, yc) * getItem(u_in, xd, yc) - getItem(h_in, xb, yc) * getItem(u_in, xb, yc)) * dx2;
        hy = (getItem(h_in, xc, yd) * getItem(v_in, xc, yd) - getItem(h_in, xc, yb) * getItem(v_in, xc, yb)) * dy2;

        // ux, uy term in momentum equation
        ux = getItem(u_in, xc, yc) * (getItem(u_in, xd, yc) - getItem(u_in, xb, yc)) * dx2;
        uy = getItem(v_in, xc, yc) * (getItem(u_in, xc, yd) - getItem(u_in, xc, yb)) * dy2;

        // vx, vy term in momentum equation
        vx = getItem(u_in, xc, yc) * (getItem(v_in, xd, yc) - getItem(v_in, xb, yc)) * dx2;
        vy = getItem(v_in, xc, yc) * (getItem(v_in, xc, yd) - getItem(v_in, xc, yb)) * dy2;

        // fu, fv term in momentum equation
        fu = getItem(f, xc, yc) * getItem(u_in, xc, yc);
        fv = getItem(f, xc, yc) * getItem(v_in, xc, yc);

        // ghx, ghy term in momentum equation
        gx = (getItem(h_in, xd, yc) - getItem(h_in, xb, yc)) * gdx2;
        gy = (getItem(h_in, xc, yd) - getItem(h_in, xc, yb)) * gdy2;

        // ku, kv term in momentum equation
        ku = vd * getItem(u_in, xc, yc);
        kv = vd * getItem(v_in, xc, yc);

        // uxx, uyy, vxx, vyy diffusion term
        uxx = (getItem(u_in, xd, yc) + getItem(u_in, xb, yc) - 2 * getItem(u_in, xc, yc)) * visdxx;
        uyy = (getItem(u_in, xc, yd) + getItem(u_in, xc, yb) - 2 * getItem(u_in, xc, yc)) * visdyy;
        vxx = (getItem(v_in, xd, yc) + getItem(v_in, xb, yc) - 2 * getItem(v_in, xc, yc)) * visdxx;
        vyy = (getItem(v_in, xc, yd) + getItem(v_in, xc, yb) - 2 * getItem(v_in, xc, yc)) * visdyy;

        // Continuity equation calculation
        h_out[i] = -(hx + hy);

        // Momentum equation calculation
        u_out[i] = -(ux + uy) + fv - gx - ku + (uxx + uyy);
        v_out[i] = -(vx + vy) - fu - gy - kv + (vxx + vyy);
    }
}
