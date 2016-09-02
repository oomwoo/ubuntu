#!/usr/bin/python
# Sanity-check image recognition using pre-trained model
# See https://github.com/oomwoo/
#
# Copyright (C) 2016 oomwoo.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3.0 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License <http://www.gnu.org/licenses/> for details.

import time
import os
import numpy as np
from PIL import Image
from neon.backends import gen_backend
from neon.layers import Affine, Conv, Pooling
from neon.models import Model
from neon.transforms import Rectlin, Softmax
from neon.initializers import Uniform
from neon.data.dataiterator import ArrayIterator


# Install Imagemagick to view images
img_size = 32

# CNN setup
home_dir = os.path.expanduser("~")
param_file_name = home_dir + "/ubuntu/model/trained_bot_model_32x32.prm"
class_names = ["forward", "left", "right", "backward"]    # from ROBOT-C bot.c
nclass = len(class_names)
be = gen_backend(backend='cpu', batch_size=1)    # NN backend
init_uni = Uniform(low=-0.1, high=0.1)           # Unnecessary NN weight initialization
bn = True                                        # enable NN batch normalization
layers = [Conv((5, 5, 16), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Pooling((2, 2)),
          Conv((3, 3, 32), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Pooling((2, 2)),
          Affine(nout=50, init=init_uni, activation=Rectlin(), batch_norm=bn),
          Affine(nout=nclass, init=init_uni, activation=Softmax())]
model = Model(layers=layers)
model.load_params(param_file_name, load_states=False)


# Load images to classify
W = img_size
H = img_size
L = W*H*3
size = H, W

def test_recognition(test_file_name):
    # Load image
    x_new = np.zeros((1, L), dtype = np.float32)
    image = Image.open(test_file_name)
    image.show()
    print("Loaded " + test_file_name)

    # Convert image to sample
    image = image.resize(size)  # , Image.ANTIALIAS)
    r, g, b = image.split()
    image = Image.merge("RGB", (b, g, r))
    image = np.asarray(image, dtype=np.float32)
    image = np.transpose(image, (2, 0, 1))
    x_new = image.reshape(1, L) - 127

    # Run neural network
    inference_set = ArrayIterator(x_new, None, nclass=nclass, lshape=(3, H, W))
    out = model.get_outputs(inference_set)
    print "Recognized as " + class_names[out.argmax()]

test_recognition(home_dir + "/ubuntu/test/forward.jpg")
test_recognition(home_dir + "/ubuntu/test/right.jpg")
test_recognition(home_dir + "/ubuntu/test/left.jpg")
test_recognition(home_dir + "/ubuntu/test/backward.jpg")
