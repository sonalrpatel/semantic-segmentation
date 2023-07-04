"""
The implementation of some helpers for path and file operations.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import numpy as np
import pandas as pd
import warnings
import csv
import os
import sys
import importlib
import xml.etree.ElementTree as xmlET
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm

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
             'prediction_path': prediction_path,
             'config_path': os.path.join(output_dir, 'config.txt')}
    
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


"""
The following are some additional implementation for files and images handling.

@Project: https://github.com/sonalrpatel/semantic-segmentation

"""
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
    ids = np.random.choice(np.arange(0, len(ls[0])), n_samples, replace=replace)
    samples = zip([np.take(l, ids) for l in ls])
    return samples, ids


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
    for index, item in tqdm(class_dict.iterrows()):
        class_names.append(item[0])
        try:
            class_labels.append(np.array([item['red'], item['green'], item['blue']]))
        except:
            try:
                class_labels.append(np.array([item['r'], item['g'], item['b']]))
            except:
                print("Column names are not appropriate")
                break

    return class_names, class_labels


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
    modules = importlib.import_module(module_name, package=module_path)

    return modules


def get_labels(modules):
    labels = modules.labels
    class_names = [labels[k].name for k in range(len(labels)) if labels[k].trainId >= 0 and labels[k].trainId < 255]    
    class_colors = [list(labels[k].color) for k in range(len(labels)) if labels[k].trainId >= 0 and labels[k].trainId < 255]

    print("Number of classes - ", len(class_colors))
    for name, color in zip(class_names, class_colors):
        print(f"{name} - {color}")

    return class_names, class_colors


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
