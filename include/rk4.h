#ifndef _RK4_H
#define _RK4_H

void rk4(float *u, float *v, float *eta, float *f);

void rk4Update(float *x, float *k1, float *k2, float *k3, float *k4);

// Calculate slope for rk4 step.
void rk4Slope(float *su, float *sv, float *sh, float *u, float *v, float *h, float *ku, float *kv, float *kh, float d);

void rk4Solve(float *u_in, float *v_in, float *h_in, float *f, float *u_out, float *v_out, float *h_out);

#endif