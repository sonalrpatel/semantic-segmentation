import numpy as np
import tensorflow as tf

backend = tf.keras.backend
losses = tf.losses


# References
# https://github.com/davidtvs/Keras-LinkNet/blob/538ec6fcaf88bb9365508c891f632f4cd42da886/metrics/miou.py
# https://gist.github.com/ilmonteux/8340df952722f3a1030a7d937e701b5a
# https://github.com/qubvel/segmentation_models/blob/94f624b7029deb463c859efbd92fa26f512b52b8/segmentation_models/base/functional.py
# https://github.com/MrGiovanni/UNetPlusPlus/blob/f8c4064659c6857d17f39088acd0d1eeb95340ea/keras/helper_functions.py


class Mean_IoU(object):
    """Mean intersection over union (mIoU) metric.
    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative)   -- (1)
        same as,
        IoU(A,B) = |A & B| / (| A U B|)                                           -- (2)
        Dice(A,B) = 2*|A & B| / (|A| + |B|)                                       -- (3)

    The mean IoU is the mean of IoU between all classes.

    Init keyword arguments:
        num_classes (int): number of classes in the segmentation problem.

    Function inputs are
        1.  B*W*H*N tensors, with
            B = batch size,
            W = width,
            H = height,
            N = number of classes
        2.  mean_per_class, for per-class metric
        3.  calc_method

    Function returns:
        IoU of y_true and y_pred, as a float, unless mean_per_class == True
        in which case it returns the per-class metric, averaged over the batch.
    """

    def __init__(self, n_classes, mean_per_class=False, conf_matrix=False, calc_method=2):
        super().__init__()
        assert (mean_per_class & conf_matrix) is False, 'Both mean_per_class and conf_matrix can not be true together'

        self.num_classes = n_classes
        self.calc_method = calc_method
        self.mean_per_class = mean_per_class
        self.conf_matrix = conf_matrix

    def mean_iou(self, y_true, y_pred):
        """The metric function to be passed to the model.
        Args:
            y_true (tensor): True labels.
            y_pred (tensor): Predictions of the same shape as y_true.

        Returns:
            The mean intersection over union as a tensor.
        """
        if self.calc_method == 1:
            return tf.py_function(self.mean_iou_1, [y_true, y_pred], tf.float32)
        else:
            return tf.py_function(self.mean_iou_2, [y_true, y_pred], tf.float32)

    def mean_iou_1(self, y_true, y_pred):
        """Computes the mean intersection over union using numpy and bincount.
        """
        # y_pred should be of 4 dimensions (B*W*H*N)
        assert len(y_pred.shape) == 4

        # First element of the shape mentions the batch_size
        batch_size = y_pred.shape[0]

        # Compute the confusion matrix to get the number of true positives, false positives, and false negatives
        # Convert targets and predictions from categorical to integer format
        # and, reshape them to size (B, W*H) = (batch_size, width * height)
        target = np.argmax(y_true, axis=-1).reshape(batch_size, -1)
        predicted = np.argmax(y_pred, axis=-1).reshape(batch_size, -1)

        # Trick for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.array([np.bincount(x[i].astype(np.int32),
                                            minlength=self.num_classes ** 2) for i in range(batch_size)])
        assert bincount_2d.shape[0] == batch_size
        assert bincount_2d.shape[1] == self.num_classes ** 2

        # Confusion matrix shape is of (B, N, N) = (batch_size, num_classes, num_classes)
        conf = bincount_2d.reshape((batch_size, self.num_classes, self.num_classes))

        # Compute the IoU and mean IoU from the confusion matrix
        # TP, FP, FN and IoU are of shape (B*N) = (batch_size * num_classes)
        true_positive = np.array([np.diag(conf[i]) for i in range(batch_size)])
        false_positive = np.sum(conf, axis=1) - true_positive
        false_negative = np.sum(conf, axis=2) - true_positive

        # Just in case we get a division by 0, ignore/hide the error and
        # set the value to 1 since we predicted 0 pixels for that class and
        # and the batch has 0 pixels for that same class
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        iou[np.isnan(iou)] = 1

        # mean per class; remaining axes are (batch, classes)
        if self.mean_per_class:
            return np.mean(iou, axis=0).astype(np.float32)

        if self.conf_matrix:
            return np.sum(conf, axis=0)

        return np.mean(iou).astype(np.float32)

    def mean_iou_2(self, y_true, y_pred):
        """Computes the mean intersection over union using keras.
        """
        # y_pred should be of 4 dimensions (B*W*H*N)
        assert len(y_pred.shape) == 4

        # Get one-hot encoded masks from y_pred (true masks should already be one-hot)
        y_true = backend.one_hot(backend.argmax(y_true), self.num_classes)
        y_pred = backend.one_hot(backend.argmax(y_pred), self.num_classes)

        # If already one-hot, could have skipped above command keras uses float32 instead of float64, would give
        # error down (but numpy arrays or keras.to_categorical gives float64)
        y_true = backend.cast(y_true, 'float32')
        y_pred = backend.cast(y_pred, 'float32')

        # Intersection and Union are of shape (B*N) = (batch_size * num_classes)
        axes = (1, 2)  # W,H axes of each image
        intersection = backend.sum(backend.abs(y_true * y_pred), axis=axes)
        mask_sum = backend.sum(backend.abs(y_true), axis=axes) + backend.sum(backend.abs(y_pred), axis=axes)
        union = mask_sum - intersection

        # IoU is of shape (B*N) = (batch_size * num_classes)
        smooth = .001
        iou = (intersection + smooth) / (union + smooth)

        # mean per class; remaining axes are (batch, classes)
        if self.mean_per_class:
            return backend.mean(iou, axis=0)

        return backend.mean(iou)
    

# Dice coefficient
def dice_coef(y_true, y_pred):
    # Get one-hot encoded masks from y_pred (true masks should already be one-hot)
    num_classes = y_pred.shape[-1]
    y_pred = backend.one_hot(backend.argmax(y_pred), num_classes)
    y_true = backend.one_hot(backend.argmax(y_true), num_classes)

    # If already one-hot, could have skipped above command keras uses float32 instead of float64, would give
    # error down (but numpy arrays or keras.to_categorical gives float64)
    y_true = backend.cast(y_true, 'float32')
    y_pred = backend.cast(y_pred, 'float32')

    # Intersection and Union are of shape (B*N) = (batch_size * num_classes)
    axes = (1, 2)  # W,H axes of each image
    smooth = .001
    intersection = backend.sum(backend.abs(y_true * y_pred), axis=axes)
    mask_sum = backend.sum(backend.abs(y_true), axis=axes) + backend.sum(backend.abs(y_pred), axis=axes)
    # union = mask_sum - intersection
    # iou = (intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (mask_sum + smooth)

    return backend.mean(dice)


# Dice coefficient
def dicecoef(y_true, y_pred):
    smooth = 0.001
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)
    return score


class LossFunc(object):
    """Calculates loss.
    """

    def __init__(self, num_classes):
        self.miou_metric = Mean_IoU(num_classes)
        super().__init__()

    def iou_loss(self, y_true, y_pred):
        loss = 1 - self.miou_metric.mean_iou(y_true, y_pred)
        return loss
    
    def dice_loss(self, y_true, y_pred):
        loss = 1 - dice_coef(y_true, y_pred)
        return loss    
    
    def CEIoU_loss(self, y_true, y_pred):
        loss = losses.categorical_crossentropy(y_true, y_pred) + self.iou_loss(y_true, y_pred)
        return loss

    def CEDice_loss(self, y_true, y_pred):
        loss = losses.categorical_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred)
        return loss
