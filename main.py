import os

from train import train
from builders.model_builder import models

losses = ['ce', 'focal_loss', 'miou_loss', 'self_balanced_focal_loss',
          'wce_loss', 'focal_loss_2', 'dice_loss_2', 'bce_dice_loss', 'tversky_loss',
          'log_cosh_dice_loss', 'jacard_loss', 'ssim_loss', 'unet3p_hybrid_loss', 'basnet_hybrid_loss']

if __name__ == "__main__":
    # run train over multiple models
    # for model in models.keys():
    for loss in losses:
        train('loss', loss)