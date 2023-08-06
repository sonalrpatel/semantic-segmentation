import random
import tensorflow as tf
from tensorflow import Tensor
from keras import backend as K
from keras import Model
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v3 import MobileNetV3Small, MobileNetV3Large
from keras.applications.regnet import RegNetX002, RegNetX004, RegNetX006, RegNetX008
from keras.applications.regnet import RegNetX016, RegNetX032, RegNetX040, RegNetX064
from keras.applications.regnet import RegNetX080, RegNetX120, RegNetX160, RegNetX320
from keras.applications.regnet import RegNetY002, RegNetY004, RegNetY006, RegNetY008
from keras.applications.regnet import RegNetY016, RegNetY032, RegNetY040, RegNetY064
from keras.applications.regnet import RegNetY080, RegNetY120, RegNetY160, RegNetY320
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2
from keras.applications.efficientnet import EfficientNetB3, EfficientNetB4, EfficientNetB5
from keras.applications.efficientnet import EfficientNetB6, EfficientNetB7
from .efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2
from .efficientnet_v2 import EfficientNetV2B3, EfficientNetV2S, EfficientNetV2M, EfficientNetV2L


backbone_names_n_layers = {
    'ResNet50': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'),
    'ResNet101': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out'),
    'ResNet152': ('conv1_relu', 'conv2_block3_out', 'conv3_block8_out', 'conv4_block36_out', 'conv5_block3_out'),
    'ResNet50V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block6_1_relu', 'post_relu'),
    'ResNet101V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block23_1_relu', 'post_relu'),
    'ResNet152V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block8_1_relu', 'conv4_block36_1_relu', 'post_relu'),
    # MobileNets
    'MobileNet' : ('conv_pw_1_relu', 'conv_pw_3_relu', 'conv_pw_5_relu', 'conv_pw_11_relu', 'conv_pw_13_relu'),
    'MobileNetV2' : ('block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'out_relu'),
    'MobileNetV3Small' : ('multiply', 're_lu_3', 'multiply_1', 'multiply_11', 'multiply_17'),
    'MobileNetV3Large' : ('re_lu_2', 're_lu_6', 'multiply_1', 'multiply_13', 'multiply_19'),
    # EfficientNet
    'EfficientNetB0': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB1': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB2': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB3': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB4': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB5': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB6': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB7': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    # EfficientNetV2
    'EfficientNetV2B0': ('block1a_project_activation', 'block2b_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetV2B1': ('block1b_add', 'block2c_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetV2B2': ('block1b_add', 'block2c_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetV2B3': ('block1b_add', 'block2c_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetV2S' : ('block1b_add', 'block2d_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetV2M' : ('block1c_add', 'block2e_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetV2L' : ('block1d_add', 'block2g_add', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    # RegNetX
    'RegNetX002' : ('regnetx002_Stage_0_XBlock_0_conv_1x1_1_relu', 'regnetx002_Stage_1_XBlock_0_conv_1x1_1_relu', 'regnetx002_Stage_2_XBlock_0_conv_1x1_1_relu', 'regnetx002_Stage_3_XBlock_0_conv_1x1_1_relu', 'regnetx002_Stage_3_XBlock_4_exit_relu'),
    'RegNetX004' : ('regnetx004_Stage_0_XBlock_0_conv_1x1_1_relu', 'regnetx004_Stage_1_XBlock_0_conv_1x1_1_relu', 'regnetx004_Stage_2_XBlock_0_conv_1x1_1_relu', 'regnetx004_Stage_3_XBlock_0_conv_1x1_1_relu', 'regnetx004_Stage_3_XBlock_4_exit_relu'),
    'RegNetX006' : ('regnetx006_Stage_0_XBlock_0_conv_1x1_1_relu', 'regnetx006_Stage_1_XBlock_0_conv_1x1_1_relu', 'regnetx006_Stage_2_XBlock_0_conv_1x1_1_relu', 'regnetx006_Stage_3_XBlock_0_conv_1x1_1_relu', 'regnetx006_Stage_3_XBlock_4_exit_relu'),
    'RegNetX008' : ('regnetx008_Stage_0_XBlock_0_conv_1x1_1_relu', 'regnetx008_Stage_1_XBlock_0_conv_1x1_1_relu', 'regnetx008_Stage_2_XBlock_0_conv_1x1_1_relu', 'regnetx008_Stage_3_XBlock_0_conv_1x1_1_relu', 'regnetx008_Stage_3_XBlock_4_exit_relu'),
    'RegNetX016' : ('regnetx016_Stage_0_XBlock_0_conv_1x1_1_relu', 'regnetx016_Stage_1_XBlock_0_conv_1x1_1_relu', 'regnetx016_Stage_2_XBlock_0_conv_1x1_1_relu', 'regnetx016_Stage_3_XBlock_0_conv_1x1_1_relu', 'regnetx016_Stage_3_XBlock_1_exit_relu'),
    'RegNetX032' : ('regnetx032_Stage_0_XBlock_0_conv_1x1_1_relu', 'regnetx032_Stage_1_XBlock_0_conv_1x1_1_relu', 'regnetx032_Stage_2_XBlock_0_conv_1x1_1_relu', 'regnetx032_Stage_3_XBlock_0_conv_1x1_1_relu', 'regnetx032_Stage_3_XBlock_1_exit_relu'),
    'RegNetX040' : ('regnetx040_Stage_0_XBlock_0_conv_1x1_1_relu', 'regnetx040_Stage_1_XBlock_0_conv_1x1_1_relu', 'regnetx040_Stage_2_XBlock_0_conv_1x1_1_relu', 'regnetx040_Stage_3_XBlock_0_conv_1x1_1_relu', 'regnetx040_Stage_3_XBlock_1_exit_relu'),
    'RegNetX064' : ('regnetx064_Stage_0_XBlock_0_conv_1x1_1_relu', 'regnetx064_Stage_1_XBlock_0_conv_1x1_1_relu', 'regnetx064_Stage_2_XBlock_0_conv_1x1_1_relu', 'regnetx064_Stage_3_XBlock_0_conv_1x1_1_relu', 'regnetx064_Stage_3_XBlock_0_exit_relu'),
    'RegNetX080' : ('regnetx080_Stage_0_XBlock_0_conv_1x1_1_relu', 'regnetx080_Stage_1_XBlock_0_conv_1x1_1_relu', 'regnetx080_Stage_2_XBlock_0_conv_1x1_1_relu', 'regnetx080_Stage_3_XBlock_0_conv_1x1_1_relu', 'regnetx080_Stage_3_XBlock_0_exit_relu'),
    'RegNetX120' : ('regnetx120_Stage_0_XBlock_0_conv_1x1_1_relu', 'regnetx120_Stage_1_XBlock_0_conv_1x1_1_relu', 'regnetx120_Stage_2_XBlock_0_conv_1x1_1_relu', 'regnetx120_Stage_3_XBlock_0_conv_1x1_1_relu', 'regnetx120_Stage_3_XBlock_0_exit_relu'),
    'RegNetX160' : ('regnetx160_Stage_0_XBlock_0_conv_1x1_1_relu', 'regnetx160_Stage_1_XBlock_0_conv_1x1_1_relu', 'regnetx160_Stage_2_XBlock_0_conv_1x1_1_relu', 'regnetx160_Stage_3_XBlock_0_conv_1x1_1_relu', 'regnetx160_Stage_3_XBlock_0_exit_relu'),
    'RegNetX320' : ('regnetx320_Stage_0_XBlock_0_conv_1x1_1_relu', 'regnetx320_Stage_1_XBlock_0_conv_1x1_1_relu', 'regnetx320_Stage_2_XBlock_0_conv_1x1_1_relu', 'regnetx320_Stage_3_XBlock_0_conv_1x1_1_relu', 'regnetx320_Stage_3_XBlock_0_exit_relu'),
    # RegNetY
    'RegNetY002' : ('regnety002_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety002_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety002_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety002_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety002_Stage_3_YBlock_4_exit_relu'),
    'RegNetY004' : ('regnety004_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety004_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety004_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety004_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety004_Stage_3_YBlock_4_exit_relu'),
    'RegNetY006' : ('regnety006_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety006_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety006_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety006_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety006_Stage_3_YBlock_3_exit_relu'),
    'RegNetY008' : ('regnety008_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety008_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety008_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety008_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety008_Stage_3_YBlock_1_exit_relu'),
    'RegNetY016' : ('regnety016_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety016_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety016_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety016_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety016_Stage_3_YBlock_1_exit_relu'),
    'RegNetY032' : ('regnety032_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety032_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety032_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety032_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety032_Stage_3_YBlock_0_exit_relu'),
    'RegNetY040' : ('regnety040_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety040_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety040_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety040_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety040_Stage_3_YBlock_0_exit_relu'),
    'RegNetY064' : ('regnety064_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety064_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety064_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety064_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety064_Stage_3_YBlock_0_exit_relu'),
    'RegNetY080' : ('regnety080_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety080_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety080_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety080_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety080_Stage_3_YBlock_0_exit_relu'),
    'RegNetY120' : ('regnety120_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety120_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety120_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety120_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety120_Stage_3_YBlock_0_exit_relu'),
    'RegNetY160' : ('regnety160_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety160_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety160_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety160_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety160_Stage_3_YBlock_0_exit_relu'),
    'RegNetY320' : ('regnety320_Stage_0_YBlock_0_conv_1x1_1_relu', 'regnety320_Stage_1_YBlock_0_conv_1x1_1_relu', 'regnety320_Stage_2_YBlock_0_conv_1x1_1_relu', 'regnety320_Stage_3_YBlock_0_conv_1x1_1_relu', 'regnety320_Stage_3_YBlock_0_exit_relu'),
}


def get_backbone(backbone_name: str,
                 input_tensor: Tensor,
                 freeze_backbone: bool,
                 unfreeze_at: str,
                 output_stride: int = None,
                 depth: int = None
                 ) -> Model:   
    if output_stride is None:
        output_stride = 32
    
    if output_stride != 32 and 'EfficientNetV2' not in backbone_name:
        raise NotImplementedError(f'output_stride other than 32 is not implemented for backbone {backbone_name}. To specify a different value for output_stride use EfficientNetV2 as network backbone.')
    
    backbone_func = eval(backbone_name)
    
    if 'EfficientNetV2' in backbone_name:
        backbone_ = backbone_func(output_stride=output_stride,
                                  include_top=False,
                                  weights='imagenet',
                                  input_tensor=input_tensor,
                                  pooling=None)
    else:
        backbone_ = backbone_func(include_top=False,
                                  weights='imagenet',
                                  input_tensor=input_tensor,
                                  pooling=None)
    
    # get backbone layer names
    layer_names = backbone_names_n_layers[backbone_name]
    
    if depth is None:
        depth = len(layer_names)

    # get the output of intermediate backbone layers to use them as skip connections
    x_skip = []
    for i in range(depth):
        x_skip.append(backbone_.get_layer(layer_names[i]).output)
        
    backbone = Model(inputs=input_tensor, outputs=x_skip, name=f'{backbone_name}_backbone')
    
    if freeze_backbone:
        backbone.trainable = False
    elif unfreeze_at is not None:
        layer_dict = {layer.name: i for i, layer in enumerate(backbone.layers)}
        unfreeze_index = layer_dict[unfreeze_at]
        for layer in backbone.layers[:unfreeze_index]:
            layer.trainable = False
    else:
        backbone.trainable = True
    
    return backbone
