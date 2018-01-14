from __future__ import print_function

import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Dropout, Activation
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam


class Unet(object):
    def __init__(self, n_class=8, dropout=0.0, batch_norm=True):

        self.n_class = n_class
        self.dropout = dropout
        self.batch_norm = batch_norm

    def generalised_dice_coef(self, y_true, y_pred, type_weight='Simple'):

        y_true = K.cast(K.reshape(y_true, K.stack([K.prod(K.shape(y_true)[:-1:]), -1])), dtype='float')
        y_pred = K.reshape(y_pred, K.stack([K.prod(K.shape(y_pred)[:-1:]), -1]))

        ref_vol = K.sum(y_true, axis=0)
        seg_vol = K.sum(y_pred, axis=0)

        if type_weight == 'Square':
            weights = tf.reciprocal(K.square(ref_vol))
        elif type_weight == 'Simple':
            weights = tf.reciprocal(ref_vol)
        elif type_weight == 'Uniform':
            weights = K.ones_like(ref_vol)
        else:
            raise ValueError("The variable type_weight \"{}\""
                             "is not defined.".format(type_weight))

        new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
        weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) * tf.reduce_max(new_weights), weights)

        intersect = K.sum(y_true * y_pred, axis=0)

        generalised_dice_numerator = 2 * tf.reduce_sum(tf.multiply(weights, intersect))
        generalised_dice_denominator = tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol))

        return generalised_dice_numerator / generalised_dice_denominator

    def generalised_dice_coef_loss(self, y_true, y_pred):
        return 1 - self.generalised_dice_coef(y_true, y_pred, type_weight='Square')

    def double_conv_layer(self, x, size, dropout):
        if K.image_dim_ordering() == 'th':
            axis = 1
        else:
            axis = 3

        conv = Conv2D(size, (3, 3), padding='same')(x)
        if self.batch_norm is True:
            conv = BatchNormalization(axis=axis)(conv)

        conv = Activation('relu')(conv)
        conv = Conv2D(size, (3, 3), padding='same')(conv)

        if self.batch_norm is True:
            conv = BatchNormalization(axis=axis)(conv)
        conv = Activation('relu')(conv)

        if dropout > 0:
            conv = Dropout(dropout)(conv)
        return conv

    def get_unet(self, img_rows=None, img_cols=None, input_channels=3):
        if K.image_dim_ordering() == 'th':
            inputs = Input((input_channels, img_rows, img_cols))
            axis = 1
        else:
            inputs = Input((img_rows, img_cols, input_channels))
            axis = 3

        filters = 32

        conv1 = self.double_conv_layer(inputs, filters, self.dropout)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.double_conv_layer(pool1, 2 * filters, self.dropout)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.double_conv_layer(pool2, 4 * filters, self.dropout)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.double_conv_layer(pool3, 8 * filters, self.dropout)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = self.double_conv_layer(pool4, 16 * filters, self.dropout)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

        conv6 = self.double_conv_layer(pool5, 32 * filters, self.dropout)

        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5], axis=axis)
        conv7 = self.double_conv_layer(up7, 16 * filters, self.dropout)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv4], axis=axis)
        conv8 = self.double_conv_layer(up8, 8 * filters, self.dropout)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=axis)
        conv9 = self.double_conv_layer(up9, 4 * filters, self.dropout)

        up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=axis)
        conv10 = self.double_conv_layer(up10, 2 * filters, self.dropout)

        up11 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=axis)
        conv11 = self.double_conv_layer(up11, filters, 0)

        conv12 = Conv2D(self.n_class, (1, 1))(conv11)
        conv12 = BatchNormalization(axis=axis)(conv12)
        conv12 = Activation('sigmoid')(conv12)

        model = Model(inputs, conv12, name="Unet")

        model.compile(optimizer=Adam(lr=1e-4), loss=self.generalised_dice_coef_loss,
                      metrics=[self.generalised_dice_coef])

        return model
