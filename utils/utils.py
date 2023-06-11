"""
The implementation of some utils.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from keras_preprocessing import image as keras_image
from PIL import Image
import sys
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as xmlET
import cv2
import glob
import os
import importlib


def abspath(path):
    abspath = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise ValueError('The path "{}" does not exist.'.format(abspath))
    return abspath


def flatten(list):
    return [item for sublist in list for item in sublist]

def get_folders_in_folder(folder):
    return [f[0] for f in os.walk(folder)][1:]

def get_files_in_folder(folder, pattern=None):
    if pattern is None:
        return sorted([os.path.join(folder, f) for f in os.listdir(folder)])
    else:
        return sorted([os.path.join(folder, f) for f in os.listdir(folder) if pattern in f])

def get_files_recursive(folder, pattern=None):
    if not bool(get_folders_in_folder(folder)):
        return get_files_in_folder(folder, pattern)
    else:
        return flatten([get_files_in_folder(f, pattern) for f in get_folders_in_folder(folder)])


def sample_list(*ls, n_samples, replace=False):
    n_samples = min(len(ls[0]), n_samples)
    idcs = np.random.choice(np.arange(0, len(ls[0])), n_samples, replace=replace)
    samples = zip([np.take(l, idcs) for l in ls])
    return samples, idcs


def load_image(filename):
    # img = Image.open(name)
    # return np.array(img)    
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_image_op(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=3)
    return img


# def resize_image(image, label, target_size=None):
#     if target_size is not None:
#         image = cv2.resize(image, dsize=target_size[::-1])
#         label = cv2.resize(label, dsize=target_size[::-1], interpolation=cv2.INTER_NEAREST)
#     return image, label


def resize_image(img, shape, interpolation=cv2.INTER_CUBIC):
    # resize relevant image axis to length of corresponding target axis while preserving aspect ratio
    axis = 0 if float(shape[0]) / float(img.shape[0]) > float(shape[1]) / float(img.shape[1]) else 1
    factor = float(shape[axis]) / float(img.shape[axis])
    img = cv2.resize(img, (0,0), fx=factor, fy=factor, interpolation=interpolation)

    # crop other image axis to match target shape
    center = img.shape[int(not axis)] / 2.0
    step = shape[int(not axis)] / 2.0
    left = int(center-step)
    right = int(center+step)
    if axis == 0:
        img = img[:, left:right]
    else:
        img = img[left:right, :]
    return img


def resize_image_op(img, fromShape, toShape, cropToPreserveAspectRatio=True, interpolation=tf.image.ResizeMethod.BICUBIC):
    if not cropToPreserveAspectRatio:
        img = tf.image.resize(img, toShape, method=interpolation)
    else:
        # first crop to match target aspect ratio
        fx = toShape[1] / fromShape[1]
        fy = toShape[0] / fromShape[0]
        relevantAxis = 0 if fx < fy else 1
        if relevantAxis == 0:
            crop = fromShape[0] * toShape[1] / toShape[0]
            img = tf.image.crop_to_bounding_box(img, 0, int((fromShape[1] - crop) / 2), fromShape[0], int(crop))
        else:
            crop = fromShape[1] * toShape[0] / toShape[1]
            img = tf.image.crop_to_bounding_box(img, int((fromShape[0] - crop) / 2), 0, int(crop), fromShape[1])

        # then resize to target shape
        img = tf.image.resize(img, toShape, method=interpolation)
    return img


def normalise_image(img):
    # 3-D numpy with mean ~= 0 and variance ~= 1
    return img / 255.


def normalise_image_op(img):
    # 3-D tensor with mean ~= 0 and variance ~= 1
    return tf.image.per_image_standardization(img)


def random_crop(image, label, crop_size):
    h, w = image.shape[0:2]
    crop_h, crop_w = crop_size

    if h < crop_h or w < crop_w:
        image = cv2.resize(image, (max(w, crop_w), max(h, crop_h)))
        label = cv2.resize(label, (max(w, crop_w), max(h, crop_h)), interpolation=cv2.INTER_NEAREST)

    h, w = image.shape[0:2]
    h_beg = np.random.randint(h - crop_h)
    w_beg = np.random.randint(w - crop_w)

    cropped_image = image[h_beg:h_beg + crop_h, w_beg:w_beg + crop_w]
    cropped_label = label[h_beg:h_beg + crop_h, w_beg:w_beg + crop_w]
    return cropped_image, cropped_label


def random_zoom(image, label, zoom_range):
    if np.ndim(label) == 2:
        label = np.expand_dims(label, axis=-1)
    assert np.ndim(label) == 3

    if np.isscalar(zoom_range):
        zx, zy = np.random.uniform(1 - zoom_range, 1 + zoom_range, 2)
    elif len(zoom_range) == 2:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    else:
        raise ValueError('`zoom_range` should be a float or '
                         'a tuple or list of two floats. '
                         'Received: %s' % (zoom_range,))

    image = keras_image.apply_affine_transform(image, zx=zx, zy=zy, fill_mode='nearest')
    label = keras_image.apply_affine_transform(label, zx=zx, zy=zy, fill_mode='nearest')
    return image, label


def random_brightness(image, label, brightness_range):
    if np.ndim(label) == 2:
        label = np.expand_dims(label, axis=-1)
    assert np.ndim(label) == 3

    if brightness_range is not None:
        if isinstance(brightness_range, (tuple, list)) and len(brightness_range) == 2:
            brightness = np.random.uniform(brightness_range[0], brightness_range[1])
        else:
            raise ValueError('`brightness_range` should be '
                             'a tuple or list of two floats. '
                             'Received: %s' % (brightness_range,))
        image = keras_image.apply_brightness_shift(image, brightness)
    return image, label


def random_horizontal_flip(image, label, h_flip):
    if h_flip:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
    return image, label


def random_vertical_flip(image, label, v_flip):
    if v_flip:
        image = cv2.flip(image, 0)
        label = cv2.flip(label, 0)
    return image, label


def random_rotation(image, label, rotation_range):
    if np.ndim(label) == 2:
        label = np.expand_dims(label, axis=-1)
    assert np.ndim(label) == 3

    if rotation_range > 0.:
        theta = np.random.uniform(-rotation_range, rotation_range)
        # rotate it!
        image = keras_image.apply_affine_transform(image, theta=theta, fill_mode='nearest')
        label = keras_image.apply_affine_transform(label, theta=theta, fill_mode='nearest')
    return image, label


def random_channel_shift(image, label, channel_shift_range):
    if np.ndim(label) == 2:
        label = np.expand_dims(label, axis=-1)
    assert np.ndim(label) == 3

    if channel_shift_range > 0:
        channel_shift_intensity = np.random.uniform(-channel_shift_range, channel_shift_range)
        image = keras_image.apply_channel_shift(image, channel_shift_intensity, channel_axis=2)
    return image, label


def one_hot_encode_gray(label, num_classes):
    if np.ndim(label) == 3 and label.shape[2] == 1:
        label = np.squeeze(label, axis=-1)
    if np.ndim(label) == 3 and label.shape[2] > 1:
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    assert np.ndim(label) == 2

    heat_map = np.ones(shape=label.shape[0:2] + (num_classes,))
    for i in range(num_classes):
        heat_map[:, :, i] = np.equal(label, i).astype('float32')
    return heat_map


def one_hot_encode_gray_op(label, num_classes):
    if len(list(label.shape)) == 3 and label.shape[2] == 1:
        label = tf.squeeze(label, axis=-1)
    if len(list(label.shape)) == 3 and label.shape[2] > 1:
        label = tf.squeeze(tf.image.rgb_to_grayscale(label), axis=-1)
    assert len(list(label.shape)) == 2

    heat_map = []
    for i in range(num_classes):
        heat_map.append(tf.equal(label, i))
    heat_map = tf.stack(heat_map, axis=-1)
    heat_map = tf.cast(heat_map, dtype=tf.float32)
    return heat_map


def one_hot_encode_label_op(image, palette):
    one_hot_map = []

    for class_colors in palette:
        class_colors = [np.array(class_colors)]
        class_map = tf.zeros(image.shape[0:2], dtype=tf.int32)
        for color in class_colors:
            # find instances of color and append layer to one-hot-map
            class_map = tf.bitwise.bitwise_or(class_map, tf.cast(tf.reduce_all(tf.equal(image, color), axis=-1), tf.int32))
        one_hot_map.append(class_map)

    # finalize one-hot-map
    one_hot_map = tf.stack(one_hot_map, axis=-1)
    one_hot_map = tf.cast(one_hot_map, tf.float32)

    return one_hot_map


def decode_one_hot(one_hot_map):
    return np.argmax(one_hot_map, axis=-1)


def parse_convert_xml(conversion_file_path):
    defRoot = xmlET.parse(conversion_file_path).getroot()

    one_hot_palette = []
    class_list = []
    for idx, defElement in enumerate(defRoot.findall("SLabel")):
        from_color = np.fromstring(defElement.get("fromColour"), dtype=int, sep=" ")
        to_class = np.fromstring(defElement.get("toValue"), dtype=int, sep=" ")
        if to_class in class_list:
             one_hot_palette[class_list.index(to_class)].append(from_color)
        else:
            one_hot_palette.append([from_color])
            class_list.append(to_class)

    return one_hot_palette


def parse_convert_py(conversion_file_path):
    module_name = os.path.splitext(os.path.basename(conversion_file_path))[0]
    module_path = os.path.abspath(os.path.expanduser(conversion_file_path))
    module_dir = os.path.dirname(module_path)
    sys.path.append(module_dir)
    module = importlib.import_module(module_name, package=module_path)

    labels = module.labels
    one_hot_palette_label_values = [list(labels[k].color) for k in range(len(labels)) if labels[k].trainId > 0 and labels[k].trainId < 255]
    one_hot_palette_label_names = [labels[k].name for k in range(len(labels)) if labels[k].trainId > 0 and labels[k].trainId < 255]

    return one_hot_palette_label_names, one_hot_palette_label_values


# adamw utils
def get_weight_decays(model, verbose=1):
    wd_dict = {}
    for layer in model.layers:
        layer_l2regs = _get_layer_l2regs(layer)
        if layer_l2regs:
            for layer_l2 in layer_l2regs:
                weight_name, weight_l2 = layer_l2
                wd_dict.update({weight_name: weight_l2})
                if weight_l2 != 0 and verbose:
                    print(("WARNING: {} l2-regularization = {} - should be "
                           "set 0 before compiling model").format(
                        weight_name, weight_l2))
    return wd_dict


def fill_dict_in_order(_dict, _list_of_vals):
    for idx, key in enumerate(_dict.keys()):
        _dict[key] = _list_of_vals[idx]
    return _dict


def _get_layer_l2regs(layer):
    if hasattr(layer, 'layer') or hasattr(layer, 'cell'):
        return _rnn_l2regs(layer)
    else:
        l2_lambda_kb = []
        for weight_name in ['kernel', 'bias']:
            _lambda = getattr(layer, weight_name + '_regularizer', None)
            if _lambda is not None:
                l2_lambda_kb.append([getattr(layer, weight_name).name,
                                     float(_lambda.l2)])
        return l2_lambda_kb


def _rnn_l2regs(layer):
    _layer = layer.layer if 'backward_layer' in layer.__dict__ else layer
    cell = _layer.cell

    l2_lambda_krb = []
    if hasattr(cell, 'kernel_regularizer') or \
            hasattr(cell, 'recurrent_regularizer') or hasattr(cell, 'bias_regularizer'):
        for weight_name in ['kernel', 'recurrent', 'bias']:
            _lambda = getattr(cell, weight_name + '_regularizer', None)
            if _lambda is not None:
                weight_name = weight_name if 'recurrent' not in weight_name \
                    else 'recurrent_kernel'
                l2_lambda_krb.append([getattr(cell, weight_name).name,
                                      float(_lambda.l2)])
    return l2_lambda_krb
