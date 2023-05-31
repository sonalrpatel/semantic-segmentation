# from utils.data_generator import ImageDataGenerator
from utils.helpers import check_related_path
from utils.callbacks import LearningRateScheduler
from utils.optimizers import *
from utils.losses import *
from utils.learning_rate import *
from utils.metrics import MeanIoU
from utils import utils
from builders import model_builder
from config.labels import labels
import tensorflow as tf
import configargparse
import cv2
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


# parse parameters from config file or CLI
parser = configargparse.ArgParser()
parser.add("-c",    "--config",     is_config_file=True,    default="config/config.semseg.cityscapes.yml", help="config file")
# parser.add("-c",    "--config",                     is_config_file=True,                        help="config file")
parser.add("-d",    "--dataset",                    type=str,       default="CamVid",           help="the path of the dataset")
parser.add("-it",   "--input_training",             type=str,       required=True,              help="directory/directories of input samples for training")
parser.add("-lt",   "--label_training",             type=str,       required=True,              help="directory of label samples for training")
parser.add("-nt",   "--max_samples_training",       type=int,       default=None,               help="maximum number of training samples")
parser.add("-iv",   "--input_validation",           type=str,       required=True,              help="directory/directories of input samples for validation")
parser.add("-lv",   "--label_validation",           type=str,       required=True,              help="directory of label samples for validation")
parser.add("-nv",   "--max_samples_validation",     type=int,       default=None,               help="maximum number of validation samples")
parser.add("-is",   "--image_shape",                type=int,       required=True, nargs=2,     help="image dimensions (HxW) of inputs and labels for network")
parser.add("-nc",   "--num_classes",                type=int,       default=32,                 help="The number of classes to be segmented")
parser.add("-ohl",  "--one-hot-palette-label",      type=str,       required=True,              help="xml-file for one-hot-conversion of labels")
parser.add("-m",    "--model",                      type=str,       required=True,              help="choose the semantic segmentation methods")
parser.add("-bm",   "--base_model",                 type=str,       default=None,               help="choose the backbone model")
parser.add("-bt",   "--batch_size",                 type=int,       default=4,                  help="training batch size")
parser.add("-bv",   "--valid_batch_size",           type=int,       default=4,                  help="validation batch size")
parser.add("-ep",   "--epochs",                     type=int,       default=40,                 help="number of epochs for training")
parser.add("-ie",   "--initial_epoch",              type=int,       default=0,                  help="initial epoch of training")
parser.add("-fc",   "--checkpoint_freq",            type=int,       default=5,                  help="epoch interval to save a model checkpoint")
parser.add("-fv",   "--validation_freq",            type=int,       default=1,                  help="how often to perform validation")
parser.add("-s",    "--data_shuffle",               type=str2bool,  default=True,               help="whether to shuffle the data")
parser.add("-rs",   "--random_seed",                type=int,       default=None,               help="random shuffle seed")
parser.add("-mw",   "--model_weights",              type=str,       default=None,               help="weights file of trained model for training continuation")
parser.add("-spe",  "--steps_per_epoch",            type=int,       default=None,               help="training steps of each epoch")
parser.add("-esp",  "--early_stopping_patience",    type=int,       default=10,                 help="patience for early-stopping due to converged validation mIoU")
parser.add("-lr",   "--learning_rate",              type=float,     default=3e-4,               help="the initial learning rate")
parser.add("-lrw",  "--lr_warmup",                  type=bool,      default=False,              help="whether to use lr warm up")
parser.add("-lrs",  "--lr_scheduler",               type=str,       default="cosine_decay",     help="strategy to schedule learning rate",
                    choices=["step_decay", "poly_decay", "cosine_decay"])
parser.add("-ls",   "--loss",                       type=str,       default=None,               help="loss function for training",
                    choices=["ce", "focal_loss", "miou_loss", "self_balanced_focal_loss"])
parser.add("-op",   "--optimizer",                  type=str,       default="adam",             help="The optimizer for training",
                    choices=["sgd", "adam", "nadam", "adamw", "nadamw", "sgdw"])
parser.add("-od",   "--output_dir",                 type=str,       required=True,              help="output directory for TensorBoard and models")

parser.add("-ar",   "--data_aug_rate",              type=float,     default=0.0,                help="the rate of data augmentation")
parser.add("-hf",   "--h_flip",                     type=str2bool,  default=False,              help="whether to randomly flip the image horizontally")
parser.add("-vf",   "--v_flip",                     type=str2bool,  default=False,              help="whether to randomly flip the image vertically")
parser.add("-rc",   "--random_crop",                type=str2bool,  default=False,              help="whether to randomly crop the image")
parser.add("-ch",   "--crop_height",                type=int,       default=256,                help="the height to crop the image")
parser.add("-cw",   "--crop_width",                 type=int,       default=256,                help="the width to crop the image")
parser.add("-rt",   "--rotation",                   type=float,     default=0.0,                help="the angle to randomly rotate the image")
parser.add("-bn",   "--brightness",                 type=float,     default=None, nargs="+",    help="randomly change the brightness (list)")
parser.add("-zr",   "--zoom_range",                 type=float,     default=0.0, nargs="+",     help="the times for zooming the image")
parser.add("-cs",   "--channel_shift",              type=float,     default=0.0,                help="the channel shift range")

conf, unknown = parser.parse_known_args()


# determine absolute filepaths
conf.input_training   = utils.abspath(conf.input_training)
conf.label_training   = utils.abspath(conf.label_training)
conf.input_validation = utils.abspath(conf.input_validation)
conf.label_validation = utils.abspath(conf.label_validation)
conf.model_weights    = utils.abspath(conf.model_weights) if conf.model_weights is not None else conf.model_weights
conf.output_dir       = utils.abspath(conf.output_dir)


# check related paths
paths = check_related_path(conf.output_dir)


# get image and label file names for training and validation
# get max_samples_training random training samples
# TODO: consider images and labels when there names matches
files_train_input = utils.get_files_recursive(conf.input_training)
files_train_label = utils.get_files_recursive(conf.label_training, "color")
_, idcs = utils.sample_list(files_train_label, n_samples=conf.max_samples_training)
files_train_input = np.take(files_train_input, idcs)
files_train_label = np.take(files_train_label, idcs)
image_shape_original_input = utils.load_image(files_train_input[0]).shape[0:2]
image_shape_original_label = utils.load_image(files_train_label[0]).shape[0:2]
print(f"Found {len(files_train_label)} training samples")

# get max_samples_validation random validation samples
files_valid_input = utils.get_files_recursive(conf.input_validation)
files_valid_label = utils.get_files_recursive(conf.label_validation, "color")
_, idcs = utils.sample_list(files_valid_label, n_samples=conf.max_samples_validation)
files_valid_input = np.take(files_valid_input, idcs)
files_valid_label = np.take(files_valid_label, idcs)
print(f"Found {len(files_valid_label)} validation samples")

# parse one-hot-conversion.xml
conf.one_hot_palette_label = utils.parse_convert_xml(conf.one_hot_palette_label)
# n_classes_label = len(conf.one_hot_palette_label)
palette_label = [[np.array(labels[k].color)] for k in range(len(labels)) if labels[k].trainId > 0 and labels[k].trainId < 255]
n_classes_label = len(palette_label)


def one_hot_encode_label_op(image, palette):
    one_hot_map = []

    for class_colors in palette:
        class_map = tf.zeros(image.shape[0:2], dtype=tf.int32)
        for color in class_colors:
            # find instances of color and append layer to one-hot-map
            class_map = tf.bitwise.bitwise_or(class_map, tf.cast(tf.reduce_all(tf.equal(image, color), axis=-1), tf.int32))
        one_hot_map.append(class_map)

    # finalize one-hot-map
    one_hot_map = tf.stack(one_hot_map, axis=-1)
    one_hot_map = tf.cast(one_hot_map, tf.float32)

    return one_hot_map

# data generator
# build dataset pipeline parsing functions
def parse_sample(input_files, label_file):
    # parse and process input images
    input = utils.load_image_op(input_files)
    input = utils.resize_image_op(input, image_shape_original_input, conf.image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # normalise the image
    input = utils.normalise_image_op(input)
    
    # parse and process label image
    label = utils.load_image_op(label_file)
    label = utils.resize_image_op(label, image_shape_original_label, conf.image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # one hot encode the seg_mask
    # label = utils.one_hot_gray_op(label, conf.num_classes)
    label = utils.one_hot_encode_label_op(label, conf.one_hot_palette_label)
    return input, label

inf = "C:/Users/pso9kor/Datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
lf = "C:/Users/pso9kor/Datasets/cityscapes/gtFine_trainvaltest/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png"

inff, lff = parse_sample(inf, lf)

print("A")