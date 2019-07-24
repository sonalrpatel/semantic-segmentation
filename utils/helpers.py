"""
The implementation of some helpers.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import warnings
import os


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


def check_related_path(current_path):
    assert os.path.exists(current_path)

    checkpoints_path = os.path.join(current_path, 'checkpoints')
    logs_path = os.path.join(checkpoints_path, 'logs')
    weights_path = os.path.join(current_path, 'weights')
    prediction_path = os.path.join(current_path, 'predictions')

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
