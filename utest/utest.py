import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)


import utils
# from utils.data_generator import DatasetGenerator
# from utils.utils import *
# from utils.helpers import *

import configargparse
import matplotlib.pyplot as plt
import numpy as np

input_training = r"C:\Users\pso9kor\Datasets\serm\0.SeRM_image\train"
label_training = r"C:\Users\pso9kor\Datasets\serm\0.SeRM_image\trainannot"
max_samples_training = 10000
label_file = r"config\serm_labels.py"
num_classes = 17
image_shape = [384, 384]
augment = True
shuffle = True
batch_size = 4
epochs = 10

# determine absolute filepaths
input_training = utils.abspath(input_training)
label_training = utils.abspath(label_training)
print(input_training, label_training)

# get class names and class colors by parsing the labels py file
class_names, class_colors, class_ids = utils.get_labels(utils.parse_convert_py(label_file))
class_ids_color = [list(np.ones(3).astype(int)*k) for k in class_ids]
assert num_classes == len(class_colors)

print("Number of classes - ", len(class_colors))
for id, name, color, id_color in zip(class_ids, class_names, class_colors, class_ids_color):
    print(f"{id} - {name} - {color} - {id_color}")

def decode_gray_index(gray_index, color_map):
    color_codes = np.array(color_map)
    pred = color_codes[gray_index.astype(int)]
    return pred

# build data pipeline
# for training
trainGen = utils.DatasetGenerator(data_path=(input_training, label_training),
                            max_samples=max_samples_training,
                            image_shape=image_shape,
                            augment=augment,
                            class_colors=class_colors,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            epochs=epochs)
dataTrain, len_train, nbatch_train = trainGen()
print("Built data pipeline for {} training samples with {} batches per epoch".format(len_train, nbatch_train))