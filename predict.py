"""
The file defines the predict process of a single RGB image.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import tensorflow as tf
from utils.helpers import check_related_path, get_colored_info, color_encode
from utils import utils
from PIL import Image
from builders import model_builder
import numpy as np
import configargparse
import sys
import cv2
import os
from models.models_segmentation import Unet, Residual_Unet, Attention_Unet, Unet_plus, DeepLabV3plus

appl = tf.keras.applications


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


parser = configargparse.ArgumentParser()
parser.add("-c",    "--config",     is_config_file=True,        default="config/config.semseg.cityscapes.yml", help="config file")
parser.add_argument('--model',                  type=str,       required=True,          help='Choose the semantic segmentation methods.')
parser.add_argument('--base_model',             type=str,       default=None,           help='Choose the backbone model.')
parser.add_argument('--csv_file',               type=str,       default=None,           help='The path of color code csv file.')
parser.add_argument('--one_hot_palette_label',  type=str,       required=True,          help="xml-file for one-hot-conversion of labels")
parser.add_argument('--num_classes',            type=int,       required=True,          help='The number of classes to be segmented.')
parser.add_argument('--image_shape',            type=int,       required=True, nargs=2, help='The image dimensions (HxW) of inputs and labels for network')
parser.add_argument('--output_dir',             type=str,       required=True,          help='The output directory for TensorBoard and models')
parser.add_argument('--weights',                type=str,       required=True,          help='The path of weights to be loaded.')
parser.add_argument('--input_testing',          type=str,       required=True,          help='The path of predicted image.')
parser.add_argument('--label_testing',          type=str,       required=True,          help='The path of predicted image.')
parser.add_argument('--max_samples_testing',    type=int,       required=True,          help='The path of predicted image.')
parser.add_argument('--color_encode',           type=str2bool,  default=True,           help='Whether to color encode the prediction.')

conf, unknown = parser.parse_known_args()

# check the image path
if not os.path.exists(conf.input_testing):
    raise ValueError('The path \'{input_testing}\' does not exist the image file.'.format(input_testing=conf.input_testing))

# build the model
# model, conf.base_model = model_builder(conf.num_classes, (conf.image_shape[0], conf.image_shape[1]), conf.model, conf.base_model)


# Instantiate Model
MODEL_TYPE = 'DeepLabV3plus' # Unet
BACKBONE = 'EfficientNetV2M'
UNFREEZE_AT = 'block6a_expand_activation' # block4a_expand_activation
INPUT_SHAPE = [256, 256, 3] # do not change
OUTPUT_STRIDE = 32
FILTERS = [16, 32, 64, 128, 256]
ACTIVATION = 'leaky_relu' # swish, leaky_relu
DROPOUT_RATE = 0
PRETRAINED_WEIGHTS = None
NUM_CLASSES = 20

model_function = eval(MODEL_TYPE)
model = model_function(input_shape=INPUT_SHAPE,
                        filters=FILTERS,
                        num_classes=NUM_CLASSES,
                        output_stride=OUTPUT_STRIDE,
                        activation=ACTIVATION,
                        dropout_rate=DROPOUT_RATE,
                        backbone_name=BACKBONE,
                        freeze_backbone=False,
                        unfreeze_at=UNFREEZE_AT,
                        )

# load weights
# check related paths
print('Loading the weights...')
if os.path.isfile(conf.weights) and os.path.exists(conf.weights):
    model.load_weights(conf.weights)

    if 'checkpoints' in os.path.dirname(conf.weights):
        output_dir = os.path.dirname(os.path.dirname(conf.weights))
    else:
        output_dir = os.path.dirname(conf.weights)

    prediction_path = os.path.join(output_dir, 'predictions')
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)
    paths = {'checkpoints_path': os.path.dirname(conf.weights),
             'prediction_path': prediction_path}
else:
    raise ValueError('The weights file does not exist in \'{path}\''.format(path=conf.weights))


# begin testing
print("\n***** Begin testing *****")
print("Model -->", conf.model)
print("Base Model -->", conf.base_model)
print("Image Shape -->", [conf.image_shape[0], conf.image_shape[1]])
print("Num Classes -->", conf.num_classes)
print("Weights Path -->", conf.weights)
print("Prediction Path -->", paths['prediction_path'])

print("")

# load_images
files_test_input = utils.get_files_recursive(conf.input_testing)
files_test_label = utils.get_files_recursive(conf.label_testing, "color")
_, idcs = utils.sample_list(files_test_label, n_samples=conf.max_samples_testing)
files_test_input = np.take(files_test_input, idcs)
files_test_label = np.take(files_test_label, idcs)

# get color info
_, color_values = utils.parse_convert_py(conf.one_hot_palette_label)

for i, name in enumerate(files_test_input.tolist()):
    sys.stdout.write('\rRunning test image %d / %d'%(i+1, len(files_test_input)))
    sys.stdout.flush()

    image = cv2.resize(utils.load_image(name), dsize=(conf.image_shape[0], conf.image_shape[1]))
    image = appl.imagenet_utils.preprocess_input(image.astype(np.float32), data_format='channels_last', mode='torch')

    # image processing
    if np.ndim(image) == 3:
        image = np.expand_dims(image, axis=0)
    assert np.ndim(image) == 4

    # get the prediction
    prediction = model.predict(image)

    if np.ndim(prediction) == 4:
        prediction = np.squeeze(prediction, axis=0)

    # decode one-hot
    prediction = utils.decode_one_hot(prediction)

    # color encode
    if conf.color_encode:
        prediction = color_encode(prediction, color_values)

    # get PIL file
    prediction = Image.fromarray(np.uint8(prediction))

    # save the prediction
    _, file_name = os.path.split(name)
    prediction.save(os.path.join(paths['prediction_path'], file_name))
