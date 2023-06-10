"""
The file defines the training process.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
# from utils.data_generator import ImageDataGenerator
from utils.helpers import check_related_path
from utils.callbacks import LearningRateScheduler
from utils.optimizers import *
from utils.learning_rate import *
from utils.metrics import MeanIoU
from utils.losses import *
from utils.lossfunc import *
from utils.loss_functions import *
from utils import utils
from builders import model_builder
import tensorflow as tf
import configargparse
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
parser.add("-d",    "--dataset",                    type=str,       default="CamVid",           help="the name of the dataset")
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
conf.one_hot_palette_label = utils.parse_convert_py(conf.one_hot_palette_label)
assert conf.num_classes == len(conf.one_hot_palette_label)

# data augmentation setting

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
    # one hot encode the label
    # label = utils.one_hot_encode_gray_op(label, conf.num_classes)
    label = utils.one_hot_encode_label_op(label, conf.one_hot_palette_label)
    return input, label


# build training data pipeline
dataTrain = tf.data.Dataset.from_tensor_slices((files_train_input, files_train_label))
dataTrain = dataTrain.shuffle(buffer_size=conf.max_samples_training, reshuffle_each_iteration=True)
dataTrain = dataTrain.map(parse_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataTrain = dataTrain.batch(conf.batch_size, drop_remainder=True)
dataTrain = dataTrain.repeat(conf.epochs)
dataTrain = dataTrain.prefetch(1)
print("Built data pipeline for training")

# build validation data pipeline
dataValid = tf.data.Dataset.from_tensor_slices((files_valid_input, files_valid_label))
dataValid = dataValid.map(parse_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataValid = dataValid.batch(conf.valid_batch_size, drop_remainder=True)
dataValid = dataValid.repeat(conf.epochs)
dataValid = dataValid.prefetch(1)
print("Built data pipeline for validation")


# build the model
model, base_model = model_builder(conf.num_classes, (conf.crop_height, conf.crop_width), conf.model, conf.base_model)

# summary
model.summary()

# load weights
if conf.model_weights is not None:
    print('Loading the weights...')
    model.load_weights(conf.model_weights)


# choose loss
losses = {'ce': categorical_crossentropy_with_logits,
          'focal_loss': focal_loss(),
          'miou_loss': miou_loss(num_classes=conf.num_classes),
          'self_balanced_focal_loss': self_balanced_focal_loss(),

          'iou_loss': LossFunc(conf.num_classes).iou_loss,
          'dice_loss': LossFunc(conf.num_classes).dice_loss,
          'ce_iou_loss': LossFunc(conf.num_classes).CEIoU_loss,
          'ce_dice_loss': LossFunc(conf.num_classes).CEDice_loss,
          
          'wce_loss': Semantic_loss_functions().weighted_cross_entropyloss,
          'focal_loss_2': Semantic_loss_functions().focal_loss,
          'dice_loss_2': Semantic_loss_functions().dice_loss,
          'bce_dice_loss': Semantic_loss_functions().bce_dice_loss,
          'tversky_loss': Semantic_loss_functions().tversky_loss,
          'log_cosh_dice_loss': Semantic_loss_functions().log_cosh_dice_loss,
          'jacard_loss': Semantic_loss_functions().jacard_loss,
          'ssim_loss': Semantic_loss_functions().ssim_loss,
          'unet3p_hybrid_loss': Semantic_loss_functions().unet3p_hybrid_loss,
          'basnet_hybrid_loss': Semantic_loss_functions().basnet_hybrid_loss}

loss = losses[conf.loss] if conf.loss is not None else categorical_crossentropy_with_logits

# chose optimizer
total_iterations = len(files_train_input) * conf.epochs // conf.batch_size
wd_dict = utils.get_weight_decays(model)
ordered_values = []
weight_decays = utils.fill_dict_in_order(wd_dict, ordered_values)

optimizers = {'adam': tf.keras.optimizers.Adam(learning_rate=conf.learning_rate),
              'nadam': tf.keras.optimizers.Nadam(learning_rate=conf.learning_rate),
              'sgd': tf.keras.optimizers.SGD(learning_rate=conf.learning_rate, momentum=0.99),
            #   'adamw': AdamW(learning_rate=conf.learning_rate, batch_size=conf.batch_size,
            #                  total_iterations=total_iterations),
            #   'nadamw': NadamW(learning_rate=conf.learning_rate, batch_size=conf.batch_size,
            #                    total_iterations=total_iterations),
            #   'sgdw': SGDW(learning_rate=conf.learning_rate, momentum=0.99, batch_size=conf.batch_size,
            #                total_iterations=total_iterations)
            }
optimizer = optimizers[conf.optimizer]

# lr schedule strategy
if conf.lr_warmup and conf.epochs - 5 <= 0:
    raise ValueError('epochs must be larger than 5 if lr warm up is used.')

lr_decays = {'step_decay': step_decay(conf.learning_rate, conf.epochs - 5 if conf.lr_warmup else conf.epochs,
                                      warmup=conf.lr_warmup),
             'poly_decay': poly_decay(conf.learning_rate, conf.epochs - 5 if conf.lr_warmup else conf.epochs,
                                      warmup=conf.lr_warmup),
             'cosine_decay': cosine_decay(conf.epochs - 5 if conf.lr_warmup else conf.epochs,
                                          conf.learning_rate, warmup=conf.lr_warmup)}
lr_decay = lr_decays[conf.lr_scheduler]

# metrics
metrics = [tf.keras.metrics.CategoricalAccuracy(), MeanIoU(conf.num_classes)]

# compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print("Compiled model *{}_based_on_{}*".format(conf.model, base_model))


# callbacks setting
# training and validation steps
steps_per_epoch = len(files_train_input) // conf.batch_size if not conf.steps_per_epoch else conf.steps_per_epoch   # n_batches_train
validation_steps = len(files_valid_input) // conf.valid_batch_size                                                  # n_batches_valid
# create callbacks to be called after each epoch
tensorboard_cb      = tf.keras.callbacks.TensorBoard(paths['logs_path'], update_freq="epoch", profile_batch=0)
csvlogger_cb        = tf.keras.callbacks.CSVLogger(os.path.join(paths['checkpoints_path'], "log.csv"), append=True, separator=',')
checkpoint_cb       = tf.keras.callbacks.ModelCheckpoint(os.path.join(paths['checkpoints_path'],
                                                                      '{model}_based_on_{base}_'.format(model=conf.model, base=base_model) + 
                                                                    #   'miou_{val_mean_io_u:04f}_' + 
                                                                      'ep_{epoch:02d}.hdf5'),
                                                         save_freq=conf.checkpoint_freq*steps_per_epoch, save_weights_only=True)
best_checkpoint_cb  = tf.keras.callbacks.ModelCheckpoint(os.path.join(paths['checkpoints_path'], 'best_weights.hdf5'),
                                                         save_best_only=True, monitor="val_mean_io_u", mode="max", save_weights_only=True)
early_stopping_cb   = tf.keras.callbacks.EarlyStopping(monitor="val_mean_io_u", mode="max", patience=conf.early_stopping_patience, verbose=1)
lr_scheduler_cb     = LearningRateScheduler(lr_decay, conf.learning_rate, conf.lr_warmup, steps_per_epoch, verbose=1)
callbacks           = [tensorboard_cb, csvlogger_cb, checkpoint_cb, best_checkpoint_cb, early_stopping_cb, lr_scheduler_cb]


# begin training
print("\n***** Begin training *****")
print("GPU -->", tf.config.list_physical_devices('GPU'))
print("Dataset -->", conf.dataset)
print("Num Images -->", len(files_train_input))
print("Model -->", conf.model)
print("Base Model -->", base_model)
print("Crop Height -->", conf.crop_height)
print("Crop Width -->", conf.crop_width)
print("Num Epochs -->", conf.epochs)
print("Initial Epoch -->", conf.initial_epoch)
print("Batch Size -->", conf.batch_size)
print("Num Classes -->", conf.num_classes)

print("")
print("Model Configuration:")
print("\tLoss -->", conf.loss)
print("\tOptimizer -->", conf.optimizer)

print("")
print("Data Augmentation:")
print("\tData Augmentation Rate -->", conf.data_aug_rate)
print("\tVertical Flip -->", conf.v_flip)
print("\tHorizontal Flip -->", conf.h_flip)
print("\tBrightness Alteration -->", conf.brightness)
print("\tRotation -->", conf.rotation)
print("\tZoom -->", conf.zoom_range)
print("\tChannel Shift -->", conf.channel_shift)

print("")


# training
model.fit(dataTrain,
          epochs=conf.epochs, initial_epoch=conf.initial_epoch, steps_per_epoch=steps_per_epoch,
          validation_data=dataValid, validation_steps=validation_steps, validation_freq=conf.validation_freq,
          # max_queue_size=10, workers=os.cpu_count(), use_multiprocessing=False,
          callbacks=callbacks)

# save weights
model.save(filepath=os.path.join(paths['weights_path'], '{model}_based_on_{base_model}.h5'.format(model=conf.model, base_model=base_model)))
