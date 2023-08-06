"""
The implementation of learning rate scheduler.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import numpy as np
import tensorflow as tf


def step_decay(lr=3e-4, max_epochs=100, warmup=False):
    """
    step decay.
    :param lr: initial lr
    :param max_epochs: max epochs
    :param warmup: warm up or not
    :return: current lr
    """
    drop = 0.1
    max_epochs = max_epochs - 5 if warmup else max_epochs

    def decay(epoch):
        lrate = lr * np.power(drop, np.floor((1 + epoch) / max_epochs))
        return lrate

    return decay


def poly_decay(lr=3e-4, max_epochs=100, warmup=False):
    """
    poly decay.
    :param lr: initial lr
    :param max_epochs: max epochs
    :param warmup: warm up or not
    :return: current lr
    """
    max_epochs = max_epochs - 5 if warmup else max_epochs

    def decay(epoch):
        lrate = lr * (1 - np.power(epoch / max_epochs, 0.9))
        return lrate

    return decay


def cosine_decay(max_lr, max_epochs=100, min_lr=1e-7, warmup=False):
    """
    cosine annealing scheduler.
    :param max_lr: max lr
    :param max_epochs: max epochs
    :param min_lr: min lr
    :param warmup: warm up or not
    :return: current lr
    """
    max_epochs = max_epochs - 5 if warmup else max_epochs

    def decay(epoch):
        lrate = min_lr + (max_lr - min_lr) * (1 + np.cos(np.pi * epoch / max_epochs)) / 2
        return lrate

    return decay


def polynomial_decay(nbatch_train):
    WEIGHT_DECAY = 0.00005
    MOMENTUM = 0.9
    START_LR = 0.01
    END_LR = 0.00001
    LR_DECAY_EPOCHS = 10
    POWER = 2

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=START_LR,
        decay_steps=LR_DECAY_EPOCHS*nbatch_train,
        end_learning_rate=END_LR,
        power=POWER,
        cycle=False,
        name=None
    )

    return lr_schedule
