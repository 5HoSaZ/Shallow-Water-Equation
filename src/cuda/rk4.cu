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

// Clamp an interger given left and right border (inclusizv).
__device__ int clampi(int value, int lb, int rb)
{
    if (value < lb)
        value = lb;
    else if (value > rb)
        value = rb;
    return value;
}

// Get array value by 2d-index (program specific).
__device__ float getItem(float *arr, int x, int y)
{
    return *(arr + x + y * nx);
}

// Reflect x-velocity on x-border.
__device__ float reflectU(float *u, int x, int y)
{
    if (x < 0 || x >= nx)
        return 0.0;
    y = clampi(y, 0, ny - 1);
    return getItem(u, x, y);
}

// Reflect y-velocity on y-border.
__device__ float reflectV(float *v, int x, int y)
{
    if (y < 0 || y >= ny)
        return 0.0;
    x = clampi(x, 0, nx - 1);
    return getItem(v, x, y);
}

// Mirror fluid's height on border.
__device__ float mirrorH(float *h, int x, int y)
{
    x = clampi(x, 0, nx - 1);
    y = clampi(y, 0, ny - 1);
    return getItem(h, x, y);
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
    int i;

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
        xc = i % nx, yc = i / nx;
        xb = xc - 1, xd = xc + 1;
        yb = yc - 1, yd = yc + 1;

        // hx, hy term in continuity equation
        hx = (mirrorH(h_in, xd, yc) * reflectU(u_in, xd, yc) - mirrorH(h_in, xb, yc) * reflectU(u_in, xb, yc)) * dx2;
        hy = (mirrorH(h_in, xc, yd) * reflectV(v_in, xc, yd) - mirrorH(h_in, xc, yb) * reflectV(v_in, xc, yb)) * dy2;

        // ux, uy term in momentum equation
        ux = getItem(u_in, xc, yc) * (reflectU(u_in, xd, yc) - reflectU(u_in, xb, yc)) * dx2;
        uy = getItem(v_in, xc, yc) * (reflectU(u_in, xc, yd) - reflectU(u_in, xc, yb)) * dy2;

        // vx, vy term in momentum equation
        vx = getItem(u_in, xc, yc) * (reflectV(v_in, xd, yc) - reflectV(v_in, xb, yc)) * dx2;
        vy = getItem(v_in, xc, yc) * (reflectV(v_in, xc, yd) - reflectV(v_in, xc, yb)) * dy2;

        // fu, fv term in momentum equation
        fu = getItem(f, xc, yc) * getItem(u_in, xc, yc);
        fv = getItem(f, xc, yc) * getItem(v_in, xc, yc);

        // ghx, ghy term in momentum equation
        gx = (mirrorH(h_in, xd, yc) - mirrorH(h_in, xb, yc)) * gdx2;
        gy = (mirrorH(h_in, xc, yd) - mirrorH(h_in, xc, yb)) * gdy2;

        // ku, kv term in momentum equation
        ku = vd * getItem(u_in, xc, yc);
        kv = vd * getItem(v_in, xc, yc);

        // uxx, uyy, vxx, vyy diffusion term
        uxx = (reflectU(u_in, xd, yc) + reflectU(u_in, xb, yc) - 2 * reflectU(u_in, xc, yc)) * visdxx;
        uyy = (reflectU(u_in, xc, yd) + reflectU(u_in, xc, yb) - 2 * reflectU(u_in, xc, yc)) * visdyy;
        vxx = (reflectV(v_in, xd, yc) + reflectV(v_in, xb, yc) - 2 * reflectV(v_in, xc, yc)) * visdxx;
        vyy = (reflectV(v_in, xc, yd) + reflectV(v_in, xc, yb) - 2 * reflectV(v_in, xc, yc)) * visdyy;

        // Continuity equation calculation
        h_out[i] = -(hx + hy);

        // Momentum equation calculation
        u_out[i] = -(ux + uy) + fv - gx - ku + (uxx + uyy);
        v_out[i] = -(vx + vy) - fu - gy - kv + (vxx + vyy);
    }
}
