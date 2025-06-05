# Shallow Water Equation - CUDA

Middle project submission for **IT4130E-Parallel & Distributed Programming, 2024.2**\
By Group 6\
Submission date: **05/06/2025**

## Group Members

| Name            | Student ID |
| --------------- | ---------- |
| Lưu Anh Đức     | 20204875   |
| Lê Thành Vinh   | 20200668   |
| Tạ Hồ Thành Đạt | 20225482   |
| Nguyễn Văn Quốc | 20214926   |
| Vũ Tùng Linh    | 20210623   |

## Project Details

This project aims to implement **Shallow Water Equation (SWE)** in C programming language and parallelize it using CUDA-GPU. Render results are created using Matplotlib-Python.

<p align="center">
    <img alt="SWE-CenterDrop" src="render\swe_CenterDrop_default.gif" width="600"/>
</p>

To generate new render data, execute either the [sequential](out\sequential) or [cuda](out\cuda) version of the SWE program. This will generate new render files for each timestep at [render\tmp](render\tmp).

```bash
# For the sequential version
out/sequential/swe.exe
# For the cuda version
out/cuda/swe.exe
```

More details about experiment configurations as well as CUDA configurations that we used for this project can be found in [settings.h](include\settings.h).

---

**Update 03/06/2025:**
You can now modify the initial pertubation by passing additional arguments to the swe.exe **(cuda version only!)**. Multiple perturbations can be passed as well.

```
cmd: out/cuda/swe.exe [perturb options] [Args]

-drop   : Create a drop at coordinate (x, y)
    Args:
    x (float): x-coordinate of drop point
    y (float): y-coordinate of drop point

-pinch  : Create a pinch at coordinate (x, y); reverse of drop
    Args:
    x (float): x-coordinate of pinch point
    y (float): y-coordinate of pinch point

-step   : Perturb using a step function with slope k (degree) and distance d
    Args:
    k (float): Step boundary slope in degree
    d (float): Distance from step boundary to centre O(0, 0)
```

New render results from different initial perturbations:

<p align="center">
    <img alt="SWE-Pinch" src="render\swe_Pinch_x=0.00_y=0.00.gif" width="45%">
    &nbsp; &nbsp;
    <img alt="SWE-DoubleDrop" src="render\swe_DoubleDrop.gif" width="45%">
</p>
<p align="center">
    <img alt="SWE-PinchDrop" src="render\swe_PinchDrop.gif" width="45%">
    &nbsp; &nbsp;
    <img alt="SWE-Step" src="render\swe_Step_k=45.0_d=2.0.gif" width="45%">
</p>

---

## Rendering

For rendering step as well as full project implementation, check out the [notebook](notebook.ipynb).
