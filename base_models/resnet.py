"""
The implementation of ResNet50/101/152 based on Tensorflow.
Some codes are based on official tensorflow source codes.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import tensorflow as tf

layers = tf.keras.layers
backend = tf.keras.backend
keras_appl = tf.keras.applications


class ResNet(object):
    def __init__(self, version='ResNet50', dilation=None, **kwargs):
        """
       The implementation of ResNet based on Tensorflow.
       :param version: 'ResNet50', 'ResNet101' or 'ResNet152'
       :param dilation: Whether to use dilation strategy
       :param kwargs: other parameters.
       """
        super(ResNet, self).__init__(**kwargs)
        params = {'ResNet50': [2, 3, 5, 2],
                  'ResNet101': [2, 3, 22, 2],
                  'ResNet152': [2, 7, 35, 2]}
        self.version = version
        assert version in params
        self.params = params[version]

        if dilation is None:
            self.dilation = [1, 1]
        else:
            self.dilation = dilation
        assert len(self.dilation) == 2

    def _identity_block(self, input_tensor, kernel_size, filters, stage, block, dilation=1):
        """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        if block > 'z':
            block = chr(ord(block) - ord('z') + ord('A') - 1)

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(filters1, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b',
                          dilation_rate=dilation)(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x

    def _conv_block(self,
                    input_tensor,
                    kernel_size,
                    filters,
                    stage,
                    block,
                    strides=(2, 2),
                    dilation=1):
        """A block that has a conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.

        # Returns
            Output tensor for the block.

        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        strides = (1, 1) if dilation > 1 else strides

        x = layers.Conv2D(filters1, (1, 1), strides=strides,
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters2, kernel_size, padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2b',
                          dilation_rate=dilation)(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters3, (1, 1),
                          kernel_initializer='he_normal',
                          name=conv_name_base + '2c')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                                 kernel_initializer='he_normal',
                                 name=conv_name_base + '1')(input_tensor)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '1')(shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def __call__(self, inputs, output_stages='c5', **kwargs):
        """
        call for ResNet50, ResNet101 or ResNet152.
        :param inputs: a 4-D tensor.
        :param output_stages: str or a list of str containing the output stages.
        :param kwargs: other parameters.
        :return: the output of different stages.
        """
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        dilation = self.dilation

        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs)
        x = layers.Conv2D(64, (7, 7),
                          strides=(2, 2),
                          padding='valid',
                          kernel_initializer='he_normal',
                          name='conv1')(x)
        x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        c1 = x

        x = self._conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        for i in range(self.params[0]):
            x = self._identity_block(x, 3, [64, 64, 256], stage=2, block=chr(ord('b') + i))
        c2 = x

        x = self._conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        for i in range(self.params[1]):
            x = self._identity_block(x, 3, [128, 128, 512], stage=3, block=chr(ord('b') + i))
        c3 = x

        x = self._conv_block(x, 3, [256, 256, 1024], stage=4, block='a', dilation=dilation[0])
        for i in range(self.params[2]):
            x = self._identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(ord('b') + i), dilation=dilation[0])
        c4 = x

        x = self._conv_block(x, 3, [512, 512, 2048], stage=5, block='a', dilation=dilation[1])
        for i in range(self.params[3]):
            x = self._identity_block(x, 3, [512, 512, 2048], stage=5, block=chr(ord('b') + i), dilation=dilation[1])
        c5 = x

        self.outputs = {'c1': c1,
                        'c2': c2,
                        'c3': c3,
                        'c4': c4,
                        'c5': c5}

        if type(output_stages) is not list:
            return self.outputs[output_stages]
        else:
            return [self.outputs[ci] for ci in output_stages]


class ResNetKA(object):
    def __init__(self, version='ResNet50KA', **kwargs):
        """
       The implementation of ResNet based on Keras.Application.
       :param version: 'ResNet50'
       :param kwargs: other parameters.
       """
        super(ResNetKA, self).__init__(**kwargs)
        self.version = version

    def __call__(self, inputs, output_stages='c5', **kwargs):
        """
        call for ResNet50.
        :param inputs: a 4-D tensor.
        :param output_stages: str or a list of str containing the output stages.
        :param kwargs: other parameters.
        :return: the output of different stages.
        """
        resnet = keras_appl.ResNet50(input_tensor=inputs, include_top=False)

        c0 = resnet.get_layer("input_1").output               # c0-inputs : - x - x 3
        c1 = resnet.get_layer("conv1_relu").output            # c1-skip   : - x - x 64
        c2 = resnet.get_layer("conv2_block3_out").output      # c2-skip   : - x - x 256
        c3 = resnet.get_layer("conv3_block4_out").output      # c3-skip   : - x - x 512
        c4 = resnet.get_layer("conv4_block6_out").output      # c4-skip   : - x - x 1024
        c5 = resnet.get_layer("conv5_block3_out").output      # c5-bridge : - x - x 2048

        self.outputs = {'c1': c1,
                        'c2': c2,
                        'c3': c3,
                        'c4': c4,
                        'c5': c5}

        if type(output_stages) is not list:
            return self.outputs[output_stages]
        else:
            return [self.outputs[ci] for ci in output_stages]
