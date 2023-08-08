"""
The model builder to build different semantic segmentation models.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from models import *
from models.models_segmentation import Unet, Residual_Unet, Attention_Unet, Unet_plus, DeepLabV3plus

models = {'FCN-8s': FCN,
          'FCN-16s': FCN,
          'FCN-32s': FCN,
          'UNet': UNet,
          'SegNet': SegNet,
          'Bayesian-SegNet': SegNet,
          'PAN': PAN,
          'PSPNet': PSPNet,
          'RefineNet': RefineNet,
          'DenseASPP': DenseASPP,
          'DeepLabV3': DeepLabV3,
          'DeepLabV3Plus': DeepLabV3Plus,
          'BiSegNet': BiSegNet}


def model_builder(input_shape: tuple, num_classes: int, model='DeepLabV3', base_model=None, pre_trained=False, freeze_backbone=False):
    assert model in models

    if model in models:
        # initialise __init__ of the selected model class
        model = models[model](input_shape, num_classes, model, base_model, pre_trained, freeze_backbone)
        
        # get the base_model name
        base_model = model.get_base_model()

        # build the model by calling __call__
        model = model()

    return model, base_model
