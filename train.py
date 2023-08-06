"""
The file defines the training process.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from utils.callbacks import LearningRateScheduler, LearningRateGetvalue
from utils.optimizers import *
from utils.learning_rate import *
from utils.metrics import MeanIoU
from utils.losses import *
from utils.helpers import *
from utils.data_generator import DatasetGenerator
from utils import utils
from builders import model_builder

import tensorflow as tf
from keras import models
from keras.optimizers import Adam, SGD, Adadelta, Nadam
from tensorflow_addons.optimizers import AdamW, SGDW, AdaBelief
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, EarlyStopping

import configargparse
import os

tf.executing_eagerly()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


# parse parameters from config file or CLI
parser = configargparse.ArgParser()
parser.add("-c",    "--config",     is_config_file=True,            default="config/config.semseg.serm.yml", help="config file")
# parser.add("-c",    "--config",                     is_config_file=True,                        help="config file")
parser.add("-lp",   "--loop_training",              type=str2bool,  default=False,              help="training in loop from main")
parser.add("-d",    "--dataset",                    type=str,       default=None,               help="the name of the dataset")
parser.add("-it",   "--input_training",             type=str,       required=True,              help="directory/directories of input samples for training")
parser.add("-lt",   "--label_training",             type=str,       required=True,              help="directory of label samples for training")
parser.add("-nt",   "--max_samples_training",       type=int,       default=None,               help="maximum number of training samples")
parser.add("-iv",   "--input_validation",           type=str,       required=True,              help="directory/directories of input samples for validation")
parser.add("-lv",   "--label_validation",           type=str,       required=True,              help="directory of label samples for validation")
parser.add("-nv",   "--max_samples_validation",     type=int,       default=None,               help="maximum number of validation samples")
parser.add("-is",   "--image_size",                 type=int,       required=True, nargs=2,     help="image dimensions (HxW) of inputs and labels for network")
parser.add("-nc",   "--num_classes",                type=int,       default=32,                 help="the number of classes to be segmented")
parser.add("-lf",   "--label_file",                 type=str,       required=True,              help="py/xml-file describing the label classes and color values")
parser.add("-m",    "--model",                      type=str,       required=True,              help="choose the semantic segmentation methods")
parser.add("-bm",   "--base_model",                 type=str,       default=None,               help="choose the base model")
parser.add("-mw",   "--model_weights",              type=str,       default=None,               help="weights file of trained model for training continuation")
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
parser.add("-lrs",  "--lr_scheduler",               type=str,       default="polynomial_decay", help="strategy to schedule learning rate")
parser.add("-ls",   "--loss",                       type=str,       default=None,               help="loss function for training")
parser.add("-op",   "--optimizer",                  type=str,       default="adam",             help="the optimizer for training")
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
    ## read CLI ##
    conf, unknown = parser.parse_known_args()
    conf.image_size = tuple(conf.image_size)

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
    conf.input_training   = abspath(conf.input_training)
    conf.label_training   = abspath(conf.label_training)
    conf.input_validation = abspath(conf.input_validation)
    conf.label_validation = abspath(conf.label_validation)
    conf.model_weights    = abspath(conf.model_weights) if conf.model_weights is not None else conf.model_weights
    conf.output_dir       = abspath(conf.output_dir)

    # check related paths
    paths = check_related_path(conf.output_dir)

    # get class names, class colors and class ids by parsing the labels py file
    class_names, class_colors, class_ids = get_labels(parse_convert_py(conf.label_file))
    class_ids_color = [list(np.ones(3).astype(int)*k) for k in class_ids]
    assert conf.num_classes == len(class_colors)

    ## build data pipeline ##
    # for training
    dataTrain, len_train, nbatch_train = DatasetGenerator(data_path=(conf.input_training, conf.label_training),
                                                          max_samples=conf.max_samples_training,
                                                          image_size=conf.image_size,
                                                          class_colors=class_ids_color,                                                          
                                                          augment=conf.augment,
                                                          shuffle=conf.shuffle,
                                                          batch_size=conf.batch_size,
                                                          epochs=conf.epochs)()
    print("Built data pipeline for {} training samples with {} batches per epoch".format(len_train, nbatch_train))

    # for validation
    dataValid, len_valid, nbatch_valid = DatasetGenerator(data_path=(conf.input_validation, conf.label_validation),
                                                          max_samples=conf.max_samples_validation,
                                                          image_size=conf.image_size,
                                                          class_colors=class_ids_color,                                                          
                                                          augment=conf.augment,
                                                          shuffle=conf.shuffle,
                                                          batch_size=conf.batch_size,
                                                          epochs=conf.epochs)()
    print("Built data pipeline for {} validation samples with {} batches per epoch".format(len_valid, nbatch_valid))


    ## build the model ##
    model, conf.base_model = model_builder(conf.image_size, conf.num_classes, conf.model, conf.base_model, pre_trained=True)

    # summary
    model.summary()

    # load weights
    if conf.model_weights is not None:
        print("Loading the weights from {}".format(conf.model_weights))
        model.load_weights(conf.model_weights)

    ## choose loss ##
    losses = {
        'ce'                    : categorical_crossentropy_with_logits,
        'focal_loss'            : focal_loss(),
        'miou_loss'             : miou_loss(num_classes=conf.num_classes),
        'self_balanced_focal_loss'  : self_balanced_focal_loss(),
        }

    loss = losses[conf.loss] if conf.loss is not None else categorical_crossentropy_with_logits

    ## chose optimizer ##
    # lr schedule
    if conf.lr_warmup and conf.epochs - 5 <= 0:
        raise ValueError('epochs must be larger than 5 if lr warm up is used.')

    lr_decays = {
        'step_decay'       : step_decay(conf.learning_rate, conf.epochs, warmup=conf.lr_warmup),
        'poly_decay'       : poly_decay(conf.learning_rate, conf.epochs, warmup=conf.lr_warmup),
        'cosine_decay'     : cosine_decay(conf.learning_rate, conf.epochs, warmup=conf.lr_warmup),
        'polynomial_decay' : polynomial_decay(nbatch_train)
        }
    lr_scheduler = lr_decays[conf.lr_scheduler]

    # optimizer
    optimizers = {
        'adam'      : Adam(learning_rate=conf.learning_rate),
        'nadam'     : Nadam(learning_rate=conf.learning_rate),
        'sgd'       : SGD(learning_rate=conf.learning_rate, momentum=0.9),
        'adamw'     : AdamW(learning_rate=conf.learning_rate, weight_decay=0.0005),
        'sgdw'      : SGDW(learning_rate=conf.learning_rate, momentum=0.9, weight_decay=0.0005),

        'Adam'      : Adam(lr_scheduler),
        'Adadelta'  : Adadelta(lr_scheduler),
        'AdamW'     : AdamW(learning_rate=lr_scheduler, weight_decay=0.0005),
        'AdaBelief' : AdaBelief(learning_rate=lr_scheduler),
        'SGDW'      : SGDW(learning_rate=lr_scheduler, weight_decay=0.0005, momentum=0.9)
        }
    optimizer = optimizers[conf.optimizer]

    ## metrics ##
    metrics = [tf.keras.metrics.CategoricalAccuracy(), MeanIoU(conf.num_classes)]

    ## compile the model ##
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("Compiled model *{}_based_on_{}*".format(conf.model, conf.base_model))

    ## callback setting ##
    # training and validation steps
    steps_per_epoch     = int(nbatch_train)
    validation_steps    = int(nbatch_valid)
    
    # create callbacks to be called after each epoch
    tensorboard_cb      = TensorBoard(paths['logs_path'], update_freq="epoch", profile_batch=0)
    csvlogger_cb        = CSVLogger(os.path.join(paths['checkpoints_path'], "log.csv"), append=True, separator=',')
    checkpoint_cb       = ModelCheckpoint(os.path.join(paths['checkpoints_path'],
                                                        '{model}_based_on_{base}_'.format(model=conf.model, base=conf.base_model) + 
                                                        # 'miou_{val_mean_io_u:04f}_' + 
                                                        'ep_{epoch:02d}.h5'),
                                                        save_freq=conf.checkpoint_freq*steps_per_epoch, save_weights_only=True)
    best_checkpoint_cb  = ModelCheckpoint(os.path.join(paths['weights_path'], "best_weights.hdf5"),
                                            save_best_only=True, monitor="val_mean_io_u", mode="max", save_weights_only=False)
    early_stopping_cb   = EarlyStopping(monitor="val_mean_io_u", mode="max", patience=conf.early_stopping_patience, verbose=1)
    lr_scheduler_cb     = LearningRateScheduler(lr_scheduler, conf.learning_rate, conf.lr_warmup, steps_per_epoch, verbose=1)
    # lr_getvalue_cb      = LearningRateGetvalue()

    # list of callbacks
    callbacks = [tensorboard_cb, csvlogger_cb, checkpoint_cb, best_checkpoint_cb, lr_scheduler_cb, early_stopping_cb]
    # callbacks = [tensorboard_cb, csvlogger_cb, checkpoint_cb, best_checkpoint_cb, lr_getvalue_cb, early_stopping_cb]


    ## begin training ##
    print("\n***** Begin training *****")
    print("GPU -->", tf.config.list_physical_devices('GPU'))
    print("Dataset -->", conf.dataset)
    print("Num Images -->", len_train)
    print("Model -->", conf.model)
    print("Base_model -->", conf.base_model)
    print("Image Shape -->", [conf.image_size[0], conf.image_size[1]])
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

    print("")

    # writing config into text file
    with open(paths['config_path'], 'w') as f:
        for key, value in vars(conf).items():
            f.write('%s:%s\n' % (key, value))

    ## training ##
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
