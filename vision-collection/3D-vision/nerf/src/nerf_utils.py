'''
General Utilities for NeRF
'''

import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

def get_translation_t(t):
    '''
    Get the translation matrix for movement in t
    '''

    matrix = [
              [1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, t],
              [0, 0, 0, 1]
    ]

    return torch.FloatTensor(matrix)

def get_rotation_phi(phi):
    '''
    Get the rotation matrix for movement in phi
    '''

    matrix = [
              [1, 0, 0, 0],
              [0, np.cos(phi), -np.sin(phi), 0],
              [0, np.sin(phi), np.cos(phi), 0],
              [0, 0, 0, 1]
    ]

    return torch.FloatTensor(matrix)

def get_rotation_theta(theta):
    '''
    Get the rotation matrix for movement in theta
    '''

    matrix = [
              [np.cos(theta), 0, -np.sin(theta), 0],
              [0, 1, 0, 0],
              [np.sin(theta), 0, np.cos(theta), 0],
              [0, 0, 0, 1]
    ]

    return torch.FloatTensor(matrix)

def pose_spherical(theta, phi, t):
    '''
    Get the camera to world matrix for the corresponding theta, phi, and t
    '''

    cam2world = get_translation_t(t)
    cam2world = get_rotation_phi(phi / 180.0 * np.pi) @ cam2world
    cam2world = get_rotation_theta(theta / 180.0 * np.pi) @ cam2world
    cam2world = torch.FloatTensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ cam2world
    return cam2world.to("cuda")

def load_data(im_hw, display_values, save_sample, white_background):
    '''
    args:
        im_hw (int): image height and width
        display_values (bool): print the shapes of images and poses, print focal value
        save_sample (bool): save a sample image
        white_background (bool): whether or not the images have white backgrounds
    return:
        images: rgb images
        train_poses: training poses
        val_poses: validation poses 
        test_poses: second 200 poses, for testing novel poses
        focal: focal length
        render_poses: different poses for rendering
    '''
    rgb_files = sorted(glob.glob("bottles/rgb/*"))
    images = np.array([np.array(Image.open(img).resize(size=(im_hw, im_hw))) for img in rgb_files])

    if white_background:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])

    pose_files = sorted(glob.glob("bottles/pose/*"))
    poses = np.array([np.array(np.loadtxt(pose_file)) for pose_file in pose_files]).astype(np.float32)
    train_val_poses = poses[:200]
    train_poses = train_val_poses[:100]
    val_poses = train_val_poses[100:200]
    test_poses = poses[200:]

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1,)[:-1]], 0).to("cuda")

    focal = np.array([875.])

    if display_values:
        print("IMAGES:", images.shape)
        print("TRAIN POSES:", train_poses.shape)
        print("VAL POSES:", val_poses.shape)
        print("TEST POSES:", test_poses.shape)
        print("FOCAL:", focal)
    
    if save_sample:
        plt.imsave("results/example.png", abs(images[0].astype(np.uint8)))

    return images, train_poses, val_poses, test_poses, focal, render_poses