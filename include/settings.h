#ifndef _SETTINGS_H
#define _SETTINGS_H

#include <math.h>

// Constants
#define PI 4 * atan(1.0)
#define g 9.8067   // Gravity.
#define omega 1e-5 // Angular rotation rate.
#define vd 1e-3    // Viscous drag coefficient.
#define vis 1.5e-5 // Fluid's kinematic viscosity.

// Experiment's configs
#define nx 101   // Number of points on the x-axis (101).
#define ny 101   // Number of points on the y-axis (101).
#define nt 10001 // Number of time steps (10001).

// CUDA configs
#define BlockSize 256
#define GridSize (nx * ny + BlockSize - 1) / BlockSize

#define x_start -5.0 // Start point for x-axis.
#define x_end 5.0    // End point for x-axis.

#define y_start -6.0 // Start point for y-axis.
#define y_end 6.0    // End point for y-axis.

#define t_start 0.0  // Start point for time.
#define t_end 1000.0 // End point for time.

#define Lx (x_end - x_start) // x-axis length.
#define Ly (y_end - y_start) // y-axis length.
#define Lt (t_end - t_start) // Time length.

#define dx Lx / (nx - 1) // x grid length.
#define dy Ly / (ny - 1) // y grid length.
#define dt Lt / (nt - 1) // Time step.

#define H0 1.0       // Base height (m).
#define perturb 0.01 // Pertubation.

// Render configs
#define renderTmp "render/tmp" // Render data save folder.
#define renderStep 1           // Renser every step.
#define renderCount (nt - 1) / renderStep + 1

#endif