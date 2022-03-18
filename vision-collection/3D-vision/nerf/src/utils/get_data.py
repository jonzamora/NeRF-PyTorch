import os
import glob
import imageio
import numpy as np

def get_data(args):

    datadir = args.datadir
    img_dir = os.path.join(datadir, "rgb/*")
    poses_dir = os.path.join(datadir, "pose/*")

    img_fnames = sorted(glob.glob(img_dir))
    train_im_names = sorted([file for file in img_fnames if "train" in file])
    val_im_names = sorted([file for file in img_fnames if "val" in file])
    val_im_names = [val_im_names[0], val_im_names[48], val_im_names[99]]
    test_im_names = sorted([file for file in img_fnames if "test" in file])

    pose_fnames = sorted(glob.glob(poses_dir))
    train_pose_names = sorted([file for file in pose_fnames if "train" in file])
    val_pose_names = sorted([file for file in pose_fnames if "val" in file])
    val_pose_names = [val_pose_names[0], val_pose_names[48], val_pose_names[99]]
    test_pose_names = ["./data/bottles/pose/2_test_0000.txt", 
                       "./data/bottles/pose/2_test_0016.txt",
                       "./data/bottles/pose/2_test_0055.txt", 
                       "./data/bottles/pose/2_test_0093.txt",
                       "./data/bottles/pose/2_test_0160.txt"]
    
    if len(test_im_names) < len(test_pose_names):
        diff = len(test_pose_names) - len(test_im_names)
        test_im_names += [None] * diff
    
    if len(val_im_names) < len(val_pose_names):
        diff = len(val_pose_names) - len(val_im_names)
        val_im_names += [None] * diff
    
    im_fnames = train_im_names + val_im_names + test_im_names
    pose_fnames = train_pose_names + val_pose_names + test_pose_names
    counts = [0, len(train_im_names), len(train_im_names) + len(val_im_names), len(im_fnames)]
    splits = [np.arange(i, j) for i, j in zip(counts, counts[1:])]
    train_split, val_split, test_split = splits
    
    images = []

    for file in im_fnames:
        if file is not None:
            image = imageio.imread(file) / 255.
            images.append(image)
    
    images = np.array(images).astype(np.float32)
    
    if args.white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    else:
        images = images[..., :3]

    poses = []

    for file in pose_fnames:
        if file is not None:
            pose = np.loadtxt(file)
            pose[:, 1:3] *= -1
            poses.append(pose.tolist())
    
    poses = np.array(poses).astype(np.float32)
    
    render_poses = np.array(poses[test_split])
    H, W = images[0].shape[:2]
    K = np.loadtxt(f"{args.datadir}/intrinsics.txt")
    focal = K[0][0]
    hwf = [H, W, focal]
    near, far = 0.0, 6.0
    
    return images, poses, render_poses, hwf, K, splits, near, far