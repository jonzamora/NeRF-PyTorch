# NeRF: Neural Radiance Fields for Novel View Synthesis

---

Written by Jonathan Zamora

## Instructions

You'll first want to create a conda environment via the following commands

```
conda env create -f environment.yaml
conda activate tdmpc
```

This will ensure you have the proper dependencies for reproducing my experiment results, otherwise, the results may not be consistent. Something else to consider is that my `environment.yaml` file is not minimal, as it contains all my package dependencies for Deep Learning experiments across several domains.

After installing dependencies, you can train a NeRF with the following command:

```
python src/train.py --config configs/bottles.txt
```

Rendering results are saved in a `logs` directory, and I use Weights and Biases to log train/val loss curves as well as renderings. The renderings include validation poses and test poses. The constraint with test poses is that we don't have access to the groundtruth images for optimization, so this is where our NeRF does novel view synthesis.

## Notes

Implementations available in both PyTorch [src](src/) and Tensorflow [notebooks](notebooks/nerf_keras.ipynb)

- PyTorch implementation based off of [NeRF-PyTorch](https://github.com/yenchenlin/nerf-pytorch)

- Tensorflow implementation based off of [3D Volumetric Rendering with NeRF](https://keras.io/examples/vision/nerf/)

- This is being tested with a private dataset named "bottles". 

Since this repository implements similar boilerplate methods as included in the NeRF-PyTorch codebase, I have abstracted away the unnecessary components of NeRF-PyTorch and created this minimal and complete version of NeRF.