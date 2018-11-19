from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import xception_fcn
from . import keras_modules_injection


@keras_modules_injection
def Xception_fcn(*args, **kwargs):
    return xception_fcn.Xception_fcn(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return xception_fcn.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return xception_fcn.preprocess_input(*args, **kwargs)
