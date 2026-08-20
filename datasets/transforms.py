import random
import torch
import torchvision.transforms.functional as TF


class PairedTransform:
    """
    Applies synchronous spatial augmentations (cropping, flips, rotations)
    to paired HR and LR images.
    """
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, hr_tensor, lr_tensor):
        """
        Args:
            hr_tensor: PyTorch tensor [C, HR_H, HR_W] in range [0, 1]
            lr_tensor: PyTorch tensor [C, LR_H, LR_W] in range [0, 1]
        Returns:
            hr_tensor, lr_tensor: Transformed PyTorch tensors
        """
        if self.is_train:
            # Synchronous random horizontal flip
            if random.random() > 0.5:
                hr_tensor = TF.hflip(hr_tensor)
                lr_tensor = TF.hflip(lr_tensor)

            # Synchronous random vertical flip
            if random.random() > 0.5:
                hr_tensor = TF.vflip(hr_tensor)
                lr_tensor = TF.vflip(lr_tensor)

            # Synchronous 90 degree rotation
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                hr_tensor = TF.rotate(hr_tensor, angle)
                lr_tensor = TF.rotate(lr_tensor, angle)

        return hr_tensor, lr_tensor
