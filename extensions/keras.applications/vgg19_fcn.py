from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import vgg19_fcn
from . import keras_modules_injection


@keras_modules_injection
def VGG19_fcn(*args, **kwargs):
    return vgg19_fcn.VGG19_fcn(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return vgg19_fcn.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return vgg19_fcn.preprocess_input(*args, **kwargs)
