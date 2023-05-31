"""
The implementation of some helpers.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import numpy as np
import pandas as pd
import warnings
import csv
import os
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm

utils = tf.keras.utils
tf_image = tf.keras.preprocessing.image

def get_dataset_info(dataset_path):
    image_label_paths = check_dataset_path(dataset_path)
    image_label_names = list()

    for i, path in enumerate(image_label_paths):
        names = list()
        if path is not None:
            files = sorted(os.listdir(path))
            for file in files:
                names.append(os.path.join(path, file))
        image_label_names.append(names)

    assert len(image_label_names[0]) == len(image_label_names[1])
    assert len(image_label_names[2]) == len(image_label_names[3])

    return image_label_names


def check_dataset_path(dataset_path):
    primary_directory = ['train', 'valid', 'test']
    secondary_directory = ['images', 'labels']

    if not os.path.exists(dataset_path):
        raise ValueError('The path of the dataset does not exist.')
    else:
        train_path = os.path.join(dataset_path, primary_directory[0])
        valid_path = os.path.join(dataset_path, primary_directory[1])
        test_path = os.path.join(dataset_path, primary_directory[2])
        if not os.path.exists(train_path):
            raise ValueError('The path of the training data does not exist.')
        if not os.path.exists(valid_path):
            raise ValueError('The path of the validation data does not exist.')
        if not os.path.exists(test_path):
            warnings.warn('The path of the test data does not exist. ')

        train_image_path = os.path.join(train_path, secondary_directory[0])
        train_label_path = os.path.join(train_path, secondary_directory[1])
        if not os.path.exists(train_image_path) or not os.path.exists(train_label_path):
            raise ValueError('The path of images or labels for training does not exist.')

        valid_image_path = os.path.join(valid_path, secondary_directory[0])
        valid_label_path = os.path.join(valid_path, secondary_directory[1])
        if not os.path.exists(valid_image_path) or not os.path.exists(valid_label_path):
            raise ValueError('The path of images or labels for validation does not exist.')

        test_image_path = os.path.join(test_path, secondary_directory[0])
        test_label_path = os.path.join(test_path, secondary_directory[1])
        if not os.path.exists(test_image_path) or not os.path.exists(test_label_path):
            warnings.warn('The path of images or labels for test does not exist.')
            test_image_path = None
            test_label_path = None

        return train_image_path, train_label_path, valid_image_path, valid_label_path, test_image_path, test_label_path


def check_related_path(output_dir):
    assert os.path.exists(output_dir)
    output_dir = os.path.join(output_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.mkdir(output_dir)

    checkpoints_path = os.path.join(output_dir, 'checkpoints')
    logs_path = os.path.join(checkpoints_path, 'logs')
    weights_path = os.path.join(output_dir, 'weights')
    prediction_path = os.path.join(output_dir, 'predictions')

    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)

    paths = {'checkpoints_path': checkpoints_path,
             'logs_path': logs_path,
             'weights_path': weights_path,
             'prediction_path': prediction_path}
    
    return paths


def get_colored_info(csv_path):
    if not os.path.exists(csv_path):
        raise ValueError('The path \'{path:}\' of csv file does not exist!'.format(path=csv_path))

    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == '.csv':
        raise ValueError('File is not a CSV!')

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csv_file:
        file_reader = csv.reader(csv_file, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
    
    return class_names, label_values


def get_evaluated_classes(file_path):
    if not os.path.exists(file_path):
        raise ValueError('The path of evaluated classes file does not exist!')

    with open(file_path, 'r') as file:
        evaluated_classes = list(map(lambda z: z.strip(), file.readlines()))

    return evaluated_classes


def color_encode(image, color_values):
    color_codes = np.array(color_values)
    x = color_codes[image.astype(int)]

    return x


# Get class names and labels
def get_class_info(class_path):
    """
    Retrieve the class names and RGB values for the selected dataset.
    Must be in CSV or XLXS format!

    # Arguments
        class_path: The file path of the class dictionary

    # Returns
        Two lists: one for the class names and the other for the class label
    """
    global class_dict
    filename, file_extension = os.path.splitext(class_path)

    if file_extension == ".csv":
        class_dict = pd.read_csv(class_path)
    elif file_extension == ".xlsx":
        class_dict = pd.read_excel(class_path)
    else:
        print("Class dictionary file format not supported")
        exit(1)

    class_names = []
    class_labels = []
    for index, item in class_dict.iterrows():
        class_names.append(item[0])
        try:
            class_labels.append(np.array([item['red'], item['green'], item['blue']]))
        except:
            try:
                class_labels.append(np.array([item['r'], item['g'], item['b']]))
            except:
                print("Column names are not appropriate")
                break

    return len(class_names), class_names, class_labels


# Get labeled segmentation mask
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


def one_hot_image(seg, class_labels):
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


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
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


def count_pixels(seg_path, class_path):
    seg_path_list = list(seg_path.glob("*.png"))
    class_names, class_labels = get_class_info(class_path)
    class_list = [str(list(x)) for x in class_labels]
    df_class_count = pd.DataFrame(columns=class_list)

    # for each segmentation mask
    for enum, seg_p in tqdm(enumerate(seg_path_list)):
        seg = tf_image.img_to_array(tf_image.load_img(seg_p, target_size=(192, 192, 3)))
        seg_2d = [str(list(x.astype(int))) for x in seg.reshape(-1, seg.shape[2])]
        unq_cls_list = [str(list(x.astype(int))) for x in np.unique(seg.reshape(-1, seg.shape[2]), axis=0)]

        df_class_count.loc[enum] = [0] * len(class_names)

        # for each unique pixel
        for unq_cls in unq_cls_list:
            df_class_count.loc[enum, unq_cls] = seg_2d.count(unq_cls)

    df_class_count.columns = class_names
    df_class_count = pd.DataFrame(df_class_count.sum(axis=0)).T

    return df_class_count


def count_unique_pixels(seg_path):
    seg_path_list = list(seg_path.glob("*.png"))

    num_unique_pixels = []

    # for each segmentation mask
    for enum, seg_p in tqdm(enumerate(seg_path_list)):
        seg = tf_image.img_to_array(tf_image.load_img(seg_p, target_size=(192, 192, 3)))
        seg_2d = [str(list(x.astype(int))) for x in seg.reshape(-1, seg.shape[2])]
        unq_cls_list = [str(list(x.astype(int))) for x in np.unique(seg.reshape(-1, seg.shape[2]), axis=0)]

        num_unique_pixels.append(len(unq_cls_list))

    return num_unique_pixels
