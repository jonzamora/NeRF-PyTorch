# NeRF: Neural Radiance Fields for Novel View Synthesis

---

Written by Jonathan Zamora



## Notes:

Implementations available in both PyTorch [src](src/) and Tensorflow [notebooks](notebooks/nerf_keras.ipynb)

- PyTorch implementation based off of [NeRF-PyTorch](https://github.com/yenchenlin/nerf-pytorch)

- Tensorflow implementation based off of [3D Volumetric Rendering with NeRF](https://keras.io/examples/vision/nerf/)

- This is being tested with a private dataset named "bottles". Since this repository implements similar boilerplate methods for NeRF in PyTorch, compared to the NeRF-PyTorch codebase, there is also functionality for testing NeRF on the more well-known datasets like "Lego".

## Pipeline

1. Load Data
    - `data_loader.py`'s `load_data(args)` calls `load_pictures(args)`, so `load_pictures(args)` is run first.
    - After `load_pictures(args)`, we run the rest of `data_loader.py`'s `load_data(args)` function

2. Preprocessing

3. Training

4. Inference
