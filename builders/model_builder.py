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

models_seg = {'Unet': Unet,
              'Residual_Unet': Residual_Unet,
              'Attention_Unet': Attention_Unet,
              'Unet_plus': Unet_plus,
              'DeepLabV3plus': DeepLabV3plus}


def model_builder(input_shape: tuple, num_classes: int, model='DeepLabV3', base_model=None, pre_trained=False):
    assert model in models or models_seg

    if model in models:
        # initialise __init__ of the selected model class
        model = models[model](input_shape, num_classes, model, base_model, pre_trained)
        
        # get the base_model name
        base_model = model.get_base_model()

        # build the model by calling __call__
        model = model()
    else:
        # instantiate model
        MODEL_TYPE = model
        BACKBONE = base_model
        UNFREEZE_AT = 'block6a_expand_activation'
        FREEZE_BACKBONE = False
        INPUT_SHAPE = input_shape + [3]
        FILTERS = [16, 32, 64, 128, 256]
        NUM_CLASSES = num_classes        
        OUTPUT_STRIDE = 32
        ACTIVATION = 'leaky_relu'
        DROPOUT_RATE = 0
        PRETRAINED_WEIGHTS = None

        model_function = eval(MODEL_TYPE)
        model = model_function(input_shape=INPUT_SHAPE,
                               filters=FILTERS,
                               num_classes=NUM_CLASSES,
                               output_stride=OUTPUT_STRIDE,
                               activation=ACTIVATION,
                               dropout_rate=DROPOUT_RATE,
                               backbone_name=BACKBONE,
                               freeze_backbone=FREEZE_BACKBONE,
                               weights=PRETRAINED_WEIGHTS
                               )

    return model, base_model
