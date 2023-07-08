"""
The implementation of Data Generator based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from typing import Any
import numpy as np
import tensorflow as tf
from utils.helpers import *
from utils.utils import *

imagenet_utils = tf.keras.applications.imagenet_utils
Iterator = tf.keras.preprocessing.image.Iterator


class DataIterator(Iterator):
    def __init__(self,
                 image_data_generator,
                 images_list,
                 labels_list,
                 num_classes,
                 batch_size,
                 target_size,
                 shuffle=True,
                 seed=None,
                 data_aug_rate=0.):
        num_images = len(images_list)

        self.image_data_generator = image_data_generator
        self.images_list = images_list
        self.labels_list = labels_list
        self.num_classes = num_classes
        self.target_size = target_size
        self.data_aug_rate = data_aug_rate

        super(DataIterator, self).__init__(num_images, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(shape=(len(index_array),) + self.target_size + (3,))
        batch_y = np.zeros(shape=(len(index_array),) + self.target_size + (self.num_classes,))

        for i, idx in enumerate(index_array):
            image, label = load_image(self.images_list[idx]), load_image(self.labels_list[idx])
            # random crop
            if self.image_data_generator.random_crop:
                image, label = random_crop(image, label, self.target_size)
            else:
                image, label = resize_image(image, label, self.target_size)
            # data augmentation
            if np.random.uniform(0., 1.) < self.data_aug_rate:
                # random vertical flip
                if np.random.randint(2):
                    image, label = random_vertical_flip(image, label, self.image_data_generator.vertical_flip)
                # random horizontal flip
                if np.random.randint(2):
                    image, label = random_horizontal_flip(image, label, self.image_data_generator.horizontal_flip)
                # random brightness
                if np.random.randint(2):
                    image, label = random_brightness(image, label, self.image_data_generator.brightness_range)
                # random rotation
                if np.random.randint(2):
                    image, label = random_rotation(image, label, self.image_data_generator.rotation_range)
                # random channel shift
                if np.random.randint(2):
                    image, label = random_channel_shift(image, label, self.image_data_generator.channel_shift_range)
                # random zoom
                if np.random.randint(2):
                    image, label = random_zoom(image, label, self.image_data_generator.zoom_range)

            image = imagenet_utils.preprocess_input(image.astype('float32'), data_format='channels_last',
                                                    mode='torch')
            label = one_hot_encode_gray(label, self.num_classes)

            batch_x[i], batch_y[i] = image, label

        return batch_x, batch_y


class ImageDataGenerator(object):
    def __init__(self,
                 random_crop=False,
                 rotation_range=0,
                 brightness_range=None,
                 zoom_range=0.0,
                 channel_shift_range=0.0,
                 horizontal_flip=False,
                 vertical_flip=False):
        self.random_crop = random_crop
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def flow(self,
             images_list,
             labels_list,
             num_classes,
             batch_size,
             target_size,
             shuffle=True,
             seed=None,
             data_aug_rate=0.):
        return DataIterator(image_data_generator=self,
                            images_list=images_list,
                            labels_list=labels_list,
                            num_classes=num_classes,
                            batch_size=batch_size,
                            target_size=target_size,
                            shuffle=shuffle,
                            seed=seed,
                            data_aug_rate=data_aug_rate)


"""
The following is for Dataset Generation using tf.data.Dataset method.

@Project: https://github.com/sonalrpatel/semantic-segmentation

"""
class DatasetGenerator(object):
    def __init__(self,
                 data_path=None,
                 max_samples=0,
                 image_shape=(256, 256),
                 augment=False,
                 class_colors=None,
                 shuffle=True,
                 batch_size=4,
                 epochs=10):
        assert max_samples != 0
        assert len(data_path) == 2
        assert len(image_shape) == 2

        image_path, label_path = data_path

        self.image_path = image_path
        self.label_path = label_path
        self.max_samples = max_samples
        self.image_shape = image_shape
        self.class_colors = class_colors
        self.augment = augment
        self.shuffle = shuffle
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = 42

    # read files from data path
    def dataset_from_path(self):
        # list files
        image_file = get_files_recursive(self.image_path)
        label_file = get_files_recursive(self.label_path, "color")
        _, ids = sample_list(label_file, n_samples=self.max_samples)
        image_file = np.take(image_file, ids)
        label_file = np.take(label_file, ids)
        data_path_ds = tf.data.Dataset.from_tensor_slices((image_file, label_file))
        
        # load files
        def load_files(image_file, label_file):
            image = load_image_op(image_file)
            label = load_image_op(label_file)
            return image, label
        
        dataset = data_path_ds.map(load_files, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset, len(dataset)

    # preprocess dataset
    def preprocess_dataset(self, dataset: tf.data.Dataset):
        # load data
        def load_data(image, label):
            image = resize_image_op(image, self.image_shape)
            label = resize_image_op(label, self.image_shape)
            
            if self.augment:
                image, label = augment(image, label)
            
            image = normalize_image_op(image)
            label = one_hot_encode_op(label, self.class_colors)
            return image, label

        dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def configure_dataset(self, dataset: tf.data.Dataset):
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.batch_size*100)
        dataset = dataset.batch(self.batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.repeat(self.epochs)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        dataset, len_dataset = self.dataset_from_path()
        dataset = self.preprocess_dataset(dataset)
        dataset = self.configure_dataset(dataset)

        # image_file = get_files_recursive(self.image_path)
        # label_file = get_files_recursive(self.label_path, "color")
        # data_path_ds = tf.data.Dataset.from_tensor_slices((image_file, label_file))

        # def load_image_resize_normalize_onehot(image_file, label_file):
        #     image = tf.image.decode_png(tf.io.read_file(image_file), channels=3)
        #     label = tf.image.decode_png(tf.io.read_file(label_file), channels=3)

        #     image = tf.image.resize(image, self.image_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        #     label = tf.image.resize(label, self.image_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        #     image = tf.cast(image, tf.float32) / 255.0
        #     label = one_hot_encode_op(label, self.class_colors)
        #     return image, label

        # dataset = (data_path_ds
        #            .cache()
        #            .map(load_image_resize_normalize_onehot, num_parallel_calls=tf.data.AUTOTUNE)
        #            .shuffle(self.batch_size * 100)
        #            .batch(self.batch_size)
        #            .repeat(self.epochs)
        #            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        #            .prefetch(buffer_size=tf.data.AUTOTUNE)
        #            )
        # len_dataset = len(dataset)
        
        nbatch_dataset = dataset.cardinality().numpy() // self.epochs
        return dataset, len_dataset, nbatch_dataset
