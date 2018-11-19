from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import vgg16_fcn
from . import keras_modules_injection


@keras_modules_injection
def VGG16_fcn(*args, **kwargs):
    return vgg16_fcn.VGG16_fcn(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return vgg16_fcn.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return vgg16_fcn.preprocess_input(*args, **kwargs)
