"""
The file defines the training process.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from utils.helpers import check_related_path
from utils.callbacks import LearningRateScheduler
from utils.optimizers import *
from utils.learning_rate import *
from utils.metrics import MeanIoU
from utils.loss_func import *
from utils.loss_functions import *
from utils.losses import *
from utils.losses_segmentation import IoULoss, DiceLoss, TverskyLoss, FocalTverskyLoss, HybridLoss, FocalHybridLoss
from utils import utils
from builders import model_builder
from models.models_segmentation import Unet, Residual_Unet, Attention_Unet, Unet_plus, DeepLabV3plus

import tensorflow as tf
from keras import models
from keras.optimizers import Adam, SGD, Adadelta, Nadam
from tensorflow_addons.optimizers import AdamW, SGDW, AdaBelief
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, EarlyStopping

import configargparse
import os

from utils.augmentations import Augment
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue, GridDropout, ColorJitter,
    RandomBrightnessContrast, RandomGamma, OneOf, Rotate, RandomSunFlare, Cutout,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, HueSaturationValue,
    RGBShift, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, OpticalDistortion, RandomSizedCrop
)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


# parse parameters from config file or CLI
parser = configargparse.ArgParser()
parser.add("-c",    "--config",     is_config_file=True,            default="config/config.semseg.cityscapes.yml", help="config file")
# parser.add("-c",    "--config",                     is_config_file=True,                        help="config file")
parser.add("-lp",   "--loop_training",              type=str2bool,  default=False,              help="training in loop from main")
parser.add("-d",    "--dataset",                    type=str,       default=None,               help="the name of the dataset")
parser.add("-it",   "--input_training",             type=str,       required=True,              help="directory/directories of input samples for training")
parser.add("-lt",   "--label_training",             type=str,       required=True,              help="directory of label samples for training")
parser.add("-nt",   "--max_samples_training",       type=int,       default=None,               help="maximum number of training samples")
parser.add("-iv",   "--input_validation",           type=str,       required=True,              help="directory/directories of input samples for validation")
parser.add("-lv",   "--label_validation",           type=str,       required=True,              help="directory of label samples for validation")
parser.add("-nv",   "--max_samples_validation",     type=int,       default=None,               help="maximum number of validation samples")
parser.add("-is",   "--image_shape",                type=int,       required=True, nargs=2,     help="image dimensions (HxW) of inputs and labels for network")
parser.add("-nc",   "--num_classes",                type=int,       default=32,                 help="the number of classes to be segmented")
parser.add("-ohl",  "--one_hot_palette_label",      type=str,       required=True,              help="xml-file for one-hot-conversion of labels")
parser.add("-m",    "--model",                      type=str,       required=True,              help="choose the semantic segmentation methods")
parser.add("-bm",   "--base_model",                 type=str,       default=None,               help="choose the base model")
parser.add("-mw",   "--model_weights",              type=str,       default=None,               help="weights file of trained model for training continuation")
parser.add("-bmw",  "--bm_weights",                 type=str,       default=None,               help="weights file of base model from pre training")
parser.add("-bt",   "--batch_size",                 type=int,       default=4,                  help="training batch size")
parser.add("-bv",   "--valid_batch_size",           type=int,       default=4,                  help="validation batch size")
parser.add("-ie",   "--epochs",                     type=int,       default=10,                 help="number of epochs of training with freezed backbone")
parser.add("-ep",   "--final_epoch",                type=int,       default=20,                 help="final epoch for training with unfreezed backbone")
parser.add("-fc",   "--checkpoint_freq",            type=int,       default=5,                  help="epoch interval to save a model checkpoint")
parser.add("-fv",   "--validation_freq",            type=int,       default=1,                  help="how often to perform validation")
parser.add("-s",    "--shuffle",                    type=str2bool,  default=True,               help="whether to shuffle the data")
parser.add("-rs",   "--random_seed",                type=int,       default=None,               help="random shuffle seed")
parser.add("-esp",  "--early_stopping_patience",    type=int,       default=10,                 help="patience for early-stopping due to converged validation mIoU")
parser.add("-lr",   "--learning_rate",              type=float,     default=3e-4,               help="the initial learning rate")
parser.add("-lrw",  "--lr_warmup",                  type=str2bool,  default=False,              help="whether to use lr warm up")
parser.add("-lrs",  "--lr_scheduler",               type=str,       default="cosine_decay",     help="strategy to schedule learning rate",
                    choices=["step_decay", "poly_decay", "cosine_decay"])
parser.add("-ls",   "--loss",                       type=str,       default=None,               help="loss function for training")
parser.add("-op",   "--optimizer",                  type=str,       default="adam",             help="The optimizer for training")
parser.add("-od",   "--output_dir",                 type=str,       required=True,              help="output directory for TensorBoard and models")

parser.add("-aug",  "--augment",                    type=str2bool,  default=False,              help="whether to perform data augmentation")
parser.add("-ar",   "--data_aug_rate",              type=float,     default=0.0,                help="the rate of data augmentation")
parser.add("-hf",   "--h_flip",                     type=str2bool,  default=False,              help="whether to randomly flip the image horizontally")
parser.add("-vf",   "--v_flip",                     type=str2bool,  default=False,              help="whether to randomly flip the image vertically")
parser.add("-rc",   "--random_crop",                type=str2bool,  default=False,              help="whether to randomly crop the image")
parser.add("-rt",   "--rotation",                   type=float,     default=0.0,                help="the angle to randomly rotate the image")
parser.add("-bn",   "--brightness",                 type=float,     default=None, nargs="+",    help="randomly change the brightness (list)")
parser.add("-zr",   "--zoom_range",                 type=float,     default=0.0, nargs="+",     help="the times for zooming the image")
parser.add("-cs",   "--channel_shift",              type=float,     default=0.0,                help="the channel shift range")


def train(*args):
    # read CLI
    conf, unknown = parser.parse_known_args()

    # check args if conf.loop_training is true
    if conf.loop_training is True:
        if len(args) != 0:
            assert len(args) == 2
            
            # update model per input from main for running train() in loop of models
            if args[0] == 'model':
                conf.model = args[1]
            # update loss per input from main for running train() in loop of losses
            if args[0] == 'loss':
                conf.loss = args[1]

    # determine absolute filepaths
    conf.input_training   = utils.abspath(conf.input_training)
    conf.label_training   = utils.abspath(conf.label_training)
    conf.input_validation = utils.abspath(conf.input_validation)
    conf.label_validation = utils.abspath(conf.label_validation)
    conf.model_weights    = utils.abspath(conf.model_weights) if conf.model_weights is not None else conf.model_weights
    conf.output_dir       = utils.abspath(conf.output_dir)

    # check related paths
    paths = check_related_path(conf.output_dir)

    # parse one-hot-conversion.xml
    _, conf.one_hot_palette_label = utils.parse_convert_py(conf.one_hot_palette_label)
    assert conf.num_classes == len(conf.one_hot_palette_label)

    def one_hot_encode_label_op(mask, color_map):
        one_hot_map = []
        for color in color_map:
            class_map = tf.zeros(mask.shape[0:2], dtype=tf.int32)
            # find instances of color and append layer to one-hot-map
            class_map = tf.bitwise.bitwise_or(class_map, tf.cast(tf.reduce_all(tf.equal(mask, color), axis=-1), tf.int32))
            one_hot_map.append(class_map)

        # finalize one-hot-map
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)
        return one_hot_map
    
    def do_augmentation(img, flip=0, mask=False):
        if mask is False:
            img = tf.image.random_brightness(img, max_delta=0.5)
            img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
            img = tf.image.random_hue(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.case([(tf.greater(flip, 0), lambda: tf.image.flip_left_right(img))], default=lambda: img)
        return img

    # data generator
    # read files from data path
    def dataset_from_path(data_path: tuple, max_samples: int):
        assert len(data_path) == 2
        assert max_samples != 0

        image_path, label_path = data_path

        # list files
        image_file = utils.get_files_recursive(image_path)
        label_file = utils.get_files_recursive(label_path, "color")
        _, ids = utils.sample_list(label_file, n_samples=max_samples)
        image_file = np.take(image_file, ids)
        label_file = np.take(label_file, ids)
        data_path_ds = tf.data.Dataset.from_tensor_slices((image_file, label_file))

        # original image & label shape
        org_shape = {'image': utils.load_image(image_file[0]).shape[0:2], 'label': utils.load_image(label_file[0]).shape[0:2]}
        
        # load files
        def load_files(image_file, label_file):
            image = utils.load_image_op(image_file)
            label = utils.load_image_op(label_file)
            return image, label
        
        dataset = data_path_ds.map(load_files, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset, len(dataset), org_shape

    # preprocess dataset
    def preprocess_dataset(dataset: tf.data.Dataset, org_shape: dict, augment: bool = False):
        # load data
        def load_data(image, label):
            image = utils.resize_image_op(image, org_shape['image'], conf.image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            label = utils.resize_image_op(label, org_shape['label'], conf.image_shape, interpolation=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            
            if augment is True:
                image = do_augmentation(image)
                label = do_augmentation(label, mask=True)
            
            image = utils.normalise_image_op(image)
            label = one_hot_encode_label_op(label, conf.one_hot_palette_label)
            return image, label

        dataset = dataset.map(load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def configure_dataset(dataset: tf.data.Dataset):
        # dataset = dataset.take(count)
        if conf.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(conf.batch_size, drop_remainder=True, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat(conf.epochs)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


    # get image and label file names for training and validation
    # get max_samples_training random training samples
    # TODO: consider images and labels when there names matches
    # print(f"Found {len(files_train_label)} training samples")

    # get max_samples_validation random validation samples
    # print(f"Found {len(files_valid_label)} validation samples")
    # build training data pipeline
    dataTrain, len_train, org_shape = dataset_from_path((conf.input_training, conf.label_training), conf.max_samples_training)
    dataTrain = preprocess_dataset(dataTrain, org_shape, conf.augment)
    dataTrain = configure_dataset(dataTrain)
    n_batches_train = dataTrain.cardinality().numpy() // conf.epochs
    print("Built data pipeline for {} training samples with {} batches per epoch".format(len_train, n_batches_train))

    # build validation data pipeline
    dataValid, len_valid, org_shape = dataset_from_path((conf.input_validation, conf.label_validation), conf.max_samples_validation)
    dataValid = preprocess_dataset(dataValid, org_shape)
    dataValid = configure_dataset(dataValid)
    n_batches_valid = dataValid.cardinality().numpy() // conf.epochs
    print("Built data pipeline for {} validation samples with {} batches per epoch".format(len_valid, n_batches_valid))


    # build the model
    # model, conf.base_model = model_builder(conf.num_classes, (conf.image_shape[0], conf.image_shape[1]), conf.model, conf.base_model,
    #                                         conf.bm_weights)

    # instantiate Model
    MODEL_TYPE = 'DeepLabV3plus' # Unet
    BACKBONE = 'EfficientNetV2M'
    UNFREEZE_AT = 'block6a_expand_activation' # block4a_expand_activation
    INPUT_SHAPE = [256, 256, 3] # do not change
    OUTPUT_STRIDE = 32
    FILTERS = [16, 32, 64, 128, 256]
    ACTIVATION = 'leaky_relu' # swish, leaky_relu
    DROPOUT_RATE = 0
    PRETRAINED_WEIGHTS = None
    NUM_CLASSES = conf.num_classes

    model_function = eval(MODEL_TYPE)
    model = model_function(input_shape=INPUT_SHAPE,
                            filters=FILTERS,
                            num_classes=NUM_CLASSES,
                            output_stride=OUTPUT_STRIDE,
                            activation=ACTIVATION,
                            dropout_rate=DROPOUT_RATE,
                            backbone_name=BACKBONE,
                            freeze_backbone=False,
                            weights=PRETRAINED_WEIGHTS
                            )

    # summary
    model.summary()

    # load weights
    if conf.model_weights is not None:
        print('Loading the weights...')
        model.load_weights(conf.model_weights)

    # choose loss
    losses = {
        #losses.py
        'ce': categorical_crossentropy_with_logits,
        'focal_loss_': focal_loss(),
        'miou_loss': miou_loss(num_classes=conf.num_classes),
        'self_balanced_focal_loss': self_balanced_focal_loss(),
        
        #losses_segmentation.py
        'FocalHybridLoss': eval('FocalHybridLoss')(),

        #loss_functions.py
        'wce_loss': Semantic_loss_functions().weighted_cross_entropyloss,
        'focal_loss': Semantic_loss_functions().focal_loss,
        'dice_loss': Semantic_loss_functions().dice_loss,
        'bce_dice_loss': Semantic_loss_functions().bce_dice_loss,
        'tversky_loss': Semantic_loss_functions().tversky_loss,
        'log_cosh_dice_loss': Semantic_loss_functions().log_cosh_dice_loss,
        'jacard_loss': Semantic_loss_functions().jacard_loss,
        'ssim_loss': Semantic_loss_functions().ssim_loss,
        'unet3p_hybrid_loss': Semantic_loss_functions().unet3p_hybrid_loss,
        'basnet_hybrid_loss': Semantic_loss_functions().basnet_hybrid_loss,
        
        #loss_func.py
        'iou_loss': LossFunc(conf.num_classes).iou_loss,
        'dice_loss_': LossFunc(conf.num_classes).dice_loss,
        'ce_iou_loss': LossFunc(conf.num_classes).CEIoU_loss,
        'ce_dice_loss': LossFunc(conf.num_classes).CEDice_loss,
    }

    loss = losses[conf.loss] if conf.loss is not None else categorical_crossentropy_with_logits

    # chose optimizer
    OPTIMIZER_NAME = conf.optimizer
    WEIGHT_DECAY = 0.00005
    MOMENTUM = 0.9
    START_LR = 0.001
    END_LR = 0.0001
    LR_DECAY_EPOCHS = 10
    POWER = 2

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=START_LR,
        decay_steps=LR_DECAY_EPOCHS*n_batches_train,
        end_learning_rate=END_LR,
        power=POWER,
        cycle=False,
        name=None
    )

    optimizers = {
        'adam'      : Adam(learning_rate=conf.learning_rate),
        'nadam'     : Nadam(learning_rate=conf.learning_rate),
        'sgd'       : SGD(learning_rate=conf.learning_rate, momentum=0.99),
        'adamw'     : AdamW(learning_rate=conf.learning_rate, weight_decay=0.00005),
        'sgdw'      : SGDW(learning_rate=conf.learning_rate, momentum=0.99, weight_decay=0.00005),

        'Adam'      : Adam(lr_schedule),
        'Adadelta'  : Adadelta(lr_schedule),
        'AdamW'     : AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY),
        'AdaBelief' : AdaBelief(learning_rate=lr_schedule),
        'SGDW'      : SGDW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    }

    optimizer = optimizers[conf.optimizer]

    # metrics
    metrics = [tf.keras.metrics.CategoricalAccuracy(), MeanIoU(conf.num_classes)]

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("Compiled model *{}_based_on_{}*".format(conf.model, conf.base_model))

    # callbacks setting
    # training and validation steps
    steps_per_epoch     = int(n_batches_train)
    validation_steps    = int(n_batches_valid)
    # create callbacks to be called after each epoch
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("\nLearning Rate is {}".format(self.model.optimizer.lr))
        
    tensorboard_cb      = TensorBoard(paths['logs_path'], update_freq="epoch", profile_batch=0)
    csvlogger_cb        = CSVLogger(os.path.join(paths['checkpoints_path'], "log.csv"), append=True, separator=',')
    checkpoint_cb       = ModelCheckpoint(os.path.join(paths['checkpoints_path'],
                                                        '{model}_based_on_{base}_'.format(model=conf.model, base=conf.base_model) + 
                                                        # 'miou_{val_mean_io_u:04f}_' + 
                                                        'ep_{epoch:02d}.h5'),
                                                        save_freq=conf.checkpoint_freq*steps_per_epoch, save_weights_only=True)
    best_checkpoint_cb  = ModelCheckpoint(os.path.join(paths['weights_path'], "weights1.hdf5"),
                                            save_best_only=True, monitor="val_mean_io_u", mode="max", save_weights_only=False)
    early_stopping_cb   = EarlyStopping(monitor="val_mean_io_u", mode="max", patience=conf.early_stopping_patience, verbose=1)
    lr_cb               = CustomCallback()

    # lr schedule strategy
    if conf.lr_warmup and conf.epochs - 5 <= 0:
        raise ValueError('epochs must be larger than 5 if lr warm up is used.')

    lr_decays = {'step_decay': step_decay(conf.learning_rate, conf.epochs, warmup=conf.lr_warmup),
                 'poly_decay': poly_decay(conf.learning_rate, conf.epochs, warmup=conf.lr_warmup),
                 'cosine_decay': cosine_decay(conf.learning_rate, conf.epochs, warmup=conf.lr_warmup)}
    lr_decay = lr_decays[conf.lr_scheduler]
    lr_scheduler_cb     = LearningRateScheduler(lr_decay, conf.learning_rate, conf.lr_warmup, steps_per_epoch, verbose=1)

    # callbacks
    # callbacks = [tensorboard_cb, csvlogger_cb, checkpoint_cb, best_checkpoint_cb, lr_scheduler_cb, early_stopping_cb, lr_cb]
    callbacks = [tensorboard_cb, csvlogger_cb, checkpoint_cb, best_checkpoint_cb, early_stopping_cb, lr_cb]


    # begin training
    print("\n***** Begin training *****")
    print("GPU -->", tf.config.list_physical_devices('GPU'))
    print("Dataset -->", conf.dataset)
    print("Num Images -->", len_train)
    print("Model -->", conf.model)
    print("Base_model -->", conf.base_model)
    print("Image Shape -->", [conf.image_shape[0], conf.image_shape[1]])
    print("Epochs -->", conf.epochs)
    print("Final_epoch -->", conf.final_epoch)
    print("Batch Size -->", conf.batch_size)
    print("Num Classes -->", conf.num_classes)

    print("")
    print("Model Configuration:")
    print("\tLoss -->", conf.loss)
    print("\tOptimizer -->", conf.optimizer)
    print("\tLr Scheduler -->", conf.lr_scheduler)

    print("")
    print("Data Augmentation:")
    print("\tData Augmentation Enable -->", conf.augment)
    # print("\tVertical Flip -->", conf.v_flip)
    # print("\tHorizontal Flip -->", conf.h_flip)
    # print("\tBrightness Alteration -->", conf.brightness)
    # print("\tRotation -->", conf.rotation)
    # print("\tZoom -->", conf.zoom_range)
    # print("\tChannel Shift -->", conf.channel_shift)

    print("")

    # writing config into text file
    with open(paths['config_path'], 'w') as f:
        for key, value in vars(conf).items():
            f.write('%s:%s\n' % (key, value))

    # training
    history = model.fit(dataTrain,
                        epochs=conf.epochs, initial_epoch=0, steps_per_epoch=steps_per_epoch,
                        validation_data=dataValid, validation_steps=validation_steps, validation_freq=conf.validation_freq,
                        # max_queue_size=10, workers=os.cpu_count(), use_multiprocessing=False,
                        callbacks=callbacks)
    
    # -------------------------- #
    if False:
        # after unfreezing the final backbone weights the batch size might need to be reduced to prevent OOM
        # re-define the dataset streams with new batch size
        # build training data pipeline
        dataTrain = tf.data.Dataset.from_tensor_slices((files_train_input, files_train_label))
        dataTrain = dataTrain.shuffle(buffer_size=conf.max_samples_training, reshuffle_each_iteration=True)
        dataTrain = dataTrain.map(parse_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataTrain = dataTrain.map(Augment(93))
        dataTrain = dataTrain.batch(conf.batch_size-2, drop_remainder=True)
        dataTrain = dataTrain.repeat(conf.final_epoch-conf.epochs)
        dataTrain = dataTrain.prefetch(1)
        n_batches_train = dataTrain.cardinality().numpy() // (conf.final_epoch-conf.epochs)
        print("Built data pipeline for training with {} batches".format(n_batches_train))

        # build validation data pipeline
        dataValid = tf.data.Dataset.from_tensor_slices((files_valid_input, files_valid_label))
        dataValid = dataValid.map(parse_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataValid = dataValid.batch(conf.valid_batch_size-2, drop_remainder=True)
        dataValid = dataValid.repeat(conf.final_epoch-conf.epochs)
        dataValid = dataValid.prefetch(1)
        n_batches_valid = dataValid.cardinality().numpy() // (conf.final_epoch-conf.epochs)
        print("Built data pipeline for validation with {} batches".format(n_batches_valid))

        # instantiate Model
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

        # load the saved weights into the model to fine tune the high level features of the feature extractor
        # fine tune the encoder network with a lower learning rate
        model.load_weights(os.path.join(paths['weights_path'], "weights1.hdf5"))
        model = models.load_model(os.path.join(paths['weights_path'], "weights1.hdf5"), compile=False)
        
        model.summary()

        # optimizer with lower learning rate
        optimizer_dict = {
            'Adam'      : Adam(END_LR),
            'Adadelta'  : Adadelta(END_LR),
            'AdamW'     : AdamW(learning_rate=END_LR, weight_decay=WEIGHT_DECAY),
            'AdaBelief' : AdaBelief(learning_rate=END_LR, weight_decay=WEIGHT_DECAY),
            'SGDW'      : SGDW(learning_rate=END_LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
        }

        optimizer = optimizer_dict[OPTIMIZER_NAME]

        # re-compile the model
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # begin training
        print("\n***** Begin training with unfreezed backbone *****")

        # training
        history = model.fit(dataTrain,
                            epochs=conf.final_epoch, initial_epoch=conf.epochs, steps_per_epoch=n_batches_train,
                            validation_data=dataValid, validation_steps=n_batches_valid, validation_freq=conf.validation_freq,
                            # max_queue_size=10, workers=os.cpu_count(), use_multiprocessing=False,
                            callbacks=callbacks)

    # save weights
    model.save(filepath=os.path.join(paths['checkpoints_path'], '{model}_based_on_{base}.h5'.format(model=conf.model, base=conf.base_model)))


if __name__ == "__main__":
    train()
