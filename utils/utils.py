"""
The implementation of some utils for loading and processing images.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import numpy as np
import cv2
import tensorflow as tf

utils = tf.keras.utils
tf_image = tf.keras.preprocessing.image


def load_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def load_image_op(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=3)
    
    return img


def resize_image(img, shape, interpolation=cv2.INTER_CUBIC):
    # resize relevant image axis to length of corresponding target axis while preserving aspect ratio
    axis = 0 if float(shape[0]) / float(img.shape[0]) > float(shape[1]) / float(img.shape[1]) else 1
    factor = float(shape[axis]) / float(img.shape[axis])
    img = cv2.resize(img, (0, 0), fx=factor, fy=factor, interpolation=interpolation)

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
    
    img = tf.cast(img, dtype=tf.uint8)
    return img


def normalize_image(img):
    return img / 255


def normalize_image_op(img):
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

    image = tf_image.apply_affine_transform(image, zx=zx, zy=zy, fill_mode='nearest')
    label = tf_image.apply_affine_transform(label, zx=zx, zy=zy, fill_mode='nearest')

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
        image = tf_image.apply_brightness_shift(image, brightness)
    
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
        image = tf_image.apply_affine_transform(image, theta=theta, fill_mode='nearest')
        label = tf_image.apply_affine_transform(label, theta=theta, fill_mode='nearest')
    
    return image, label


def random_channel_shift(image, label, channel_shift_range):
    if np.ndim(label) == 2:
        label = np.expand_dims(label, axis=-1)
    assert np.ndim(label) == 3

    if channel_shift_range > 0:
        channel_shift_intensity = np.random.uniform(-channel_shift_range, channel_shift_range)
        image = tf_image.apply_channel_shift(image, channel_shift_intensity, channel_axis=2)
    
    return image, label


def do_augmentation(img, seed, mask=False):
    img = tf.image.random_flip_left_right(img, seed=seed)
    if mask is False:
        img = tf.image.random_brightness(img, max_delta=0.5, seed=seed)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5, seed=seed)
        img = tf.image.random_jpeg_quality(img, min_jpeg_quality=75, max_jpeg_quality=95, seed=seed)
        # img = tf.image.stateless_random_brightness(img, max_delta=0.5, seed=seed)
        # img = tf.image.stateless_random_contrast(img, lower=0.5, upper=1.5, seed=seed)
        # img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=75, max_jpeg_quality=95, seed=seed)
    return img


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


def label_segmentation_mask(seg, class_labels):
    """
    Given a 3D (W, H, depth=3) segmentation mask, prepare a 2D labeled segmentation mask

    # Arguments
        seg: The segmentation mask where each cell of depth provides the r, g, and b values
        class_labels

    # Returns
        Labeled segmentation mask where each cell provides its label value
    """
    seg = seg.astype("uint8")

    # returns a 2D matrix of size W x H of the segmentation mask
    label = np.zeros(seg.shape[:2], dtype=np.uint8)

    for i, rgb in enumerate(class_labels):
        label[(seg == rgb).all(axis=2)] = i

    return label


def one_hot_encode(seg, class_labels):
    """
    Convert a segmentation mask label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        seg: The 3D array segmentation mask
        class_labels

    # Returns
        A 3D array with the same width and height as the input, but
        with a depth size of num_classes
    """
    num_classes = len(class_labels)  # seg dim = H*W*3
    label = label_segmentation_mask(seg, class_labels)  # label dim = H*W
    one_hot = utils.to_categorical(label, num_classes)  # one_hot dim = H*W*N

    return one_hot


def one_hot_encode_op(mask, class_colors):
    one_hot_map = []

    for color in class_colors:
        class_map = tf.zeros(mask.shape[0:2], dtype=tf.int32)
        # find instances of color and append layer to one-hot-map
        class_map = tf.bitwise.bitwise_or(class_map, tf.cast(tf.reduce_all(tf.equal(mask, color), axis=-1), tf.int32))
        one_hot_map.append(class_map)

    # finalize one-hot-map
    one_hot_map = tf.stack(one_hot_map, axis=-1)
    one_hot_map = tf.cast(one_hot_map, tf.float32)

    return one_hot_map


# def one_hot_encode_op(image, palette):
#     one_hot_map = []
#     for class_colors in palette:
#         class_colors = [np.array(class_colors)]
#         class_map = tf.zeros(image.shape[0:2], dtype=tf.int32)
#         for color in class_colors:
#             # find instances of color and append layer to one-hot-map
#             class_map = tf.bitwise.bitwise_or(class_map, tf.cast(tf.reduce_all(tf.equal(image, color), axis=-1), tf.int32))
#         one_hot_map.append(class_map)
#     # finalize one-hot-map
#     one_hot_map = tf.stack(one_hot_map, axis=-1)
#     one_hot_map = tf.cast(one_hot_map, tf.float32)
#     return one_hot_map


def decode_one_hot_gray(one_hot_map):
    return np.argmax(one_hot_map, axis=-1)


def decode_one_hot(one_hot_map, color_map):
    pred = np.argmax(one_hot_map, axis=-1)
    color_codes = np.array(color_map)
    pred = color_codes[pred.astype(int)]
    return pred


def decode_one_hot_op(one_hot_map, color_map):
    pred = tf.argmax(one_hot_map, axis=-1)
    pred = tf.convert_to_tensor(np.array(color_map)[tf.cast(pred, tf.int8)])
    return pred


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and height as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    w = image.shape[0]
    h = image.shape[1]
    x = np.zeros([w, h, 1])

    for i in range(0, w):
        for j in range(0, h):
            index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
            x[i, j] = index

    x = np.argmax(image, axis=-1)

    return x


def make_prediction(model, img=None, img_path=None, shape=None):
    """
    Predict the hot encoded categorical label from the image.
    Later, convert it numerical label.
    """
    if img is not None:  # dim = H*W*3
        img = np.expand_dims(img, axis=0)  # dim = 1*H*W*3
    if img_path is not None:
        img = tf_image.img_to_array(tf_image.load_img(img_path, target_size=shape)) / 255.
        img = np.expand_dims(img, axis=0)  # dim = 1*H*W*3
    label = model.predict(img)  # dim = 1*H*W*N
    label = np.argmax(label[0], axis=2)  # dim = H*W

    return label


def form_color_mask(label, mapping):
    """
    Generate the color mask from the numerical label
    """
    h, w = label.shape  # dim = H*W
    mask = np.zeros((h, w, 3), dtype=np.uint8)  # dim = H*W*3
    mask = mapping[label]
    mask = mask.astype(np.uint8)

    return mask


def color_encode(image, color_values):
    color_codes = np.array(color_values)
    x = color_codes[image.astype(int)]

    return x
