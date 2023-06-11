"""
The model builder to build different semantic segmentation models.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from models import *

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


def model_builder(num_classes, input_size=(256, 256), model='SegNet', base_model=None):
    assert model in models
    assert isinstance(input_size, tuple)

    # initialise the selected model class
    model = models[model](num_classes, input_size, model, base_model)
    
    # get the base_model name
    base_model = model.get_base_model()

    # build the model by calling __call__
    model = model()

    return model, base_model
