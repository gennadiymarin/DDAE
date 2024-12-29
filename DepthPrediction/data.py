import numpy as np
from torch.utils.data import Dataset
import h5py
from PIL import Image



class NYUDepthDataset(Dataset):
    def __init__(self, images, depths, transform_rgb=None, transform_depth=None):
        """
        Args:
            images (numpy array): RGB images of shape (N, H, W, 3).
            depths (numpy array): Depth maps of shape (N, H, W).
            transform_rgb (callable, optional): Transformations for RGB images.
            transform_depth (callable, optional): Transformations for depth maps.
        """
        self.images = images
        self.depths = depths
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        rgb = Image.fromarray(self.images[idx])
        depth = Image.fromarray(self.depths[idx])

        if self.transform_rgb:
            rgb = self.transform_rgb(rgb)
        if self.transform_depth:
            depth = self.transform_depth(depth)

        return rgb, depth

def get_dataset(path, transforms_rgb=None, transforms_depth=None):
  with h5py.File(path, 'r') as f:
    images = np.array(f['images'])  # Shape: (3, H, W, N)
    images = np.transpose(images, (0, 3, 2, 1))  #(N, H, W, 3)

    depths = np.array(f['depths'])  # Shape: (H, W, N)
    depths = np.transpose(depths, (0, 2, 1))  # (N, H, W)

  nyu_dataset =  NYUDepthDataset(images, depths, transforms_rgb, transforms_depth)
  return nyu_dataset