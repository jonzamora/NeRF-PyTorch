'''
Train a NeRF model
'''

import torch
from nerf_utils import *
from model import *

def main():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    images, train_poses, val_poses, test_poses, focal, render_poses = load_data(im_hw=100, display_values=True, 
                                                                                save_sample=True, white_background=True)
    
    near, far = 1.0, 5.0

    render_kwargs_train, render_kwargs_test, start, grad_params, optimizer = build_nerf()

    global_step = start

    bounds = {"near": near, "far": far}
    render_kwargs_train.update(bounds)
    render_kwargs_test.update(bounds)

    n_rand = 1024

    train_poses = torch.Tensor(train_poses).to("cuda")
    val_poses = torch.Tensor(val_poses).to("cuda")

if __name__ == "__main__":
    main()