#!/usr/bin/python
# Connect Raspberry Pi to VEX Cortex using bi-directional UART serial link
# and exchange certain control commands
# This code runs on Raspberry Pi.
# VEX Cortex must be running peer code for the link to operate,
# see https://github.com/oomwoo/vex
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

import time, os
import numpy as np
from PIL import Image
from neon.util.argparser import NeonArgparser
from neon.backends import gen_backend
from neon.layers import Affine, Conv, Pooling, GeneralizedCost
from neon.models import Model
from neon.transforms import Rectlin, Softmax
from neon.initializers import Uniform
from neon.data.dataiterator import ArrayIterator

# CNN setup
W = 64
H = W
home_dir = os.path.expanduser("~")
# test_file_name = home_dir + "/ubuntu/test/forward.jpg"
# test_file_name = home_dir + "/ubuntu/test/backward.jpg"
test_file_name = home_dir + "/ubuntu/test/left.jpg"
param_file_name = home_dir + "/ubuntu/model/trained_bot_model_debug10.prm"
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
L = W*H*3
size = H, W

# Load an image to classify
image = Image.open(test_file_name)
print("Loaded " + test_file_name)
start_time = time.time()
image = image.resize(size)  # , Image.ANTIALIAS)
image = np.asarray(image, dtype=np.float32)

# Run neural network
x_new = image.reshape(1, L) / 255
inference_set = ArrayIterator(x_new, None, nclass=nclass, lshape=(3, H, W))
out = model.get_outputs(inference_set)
print("--- %s seconds per decision --- " % (time.time() - start_time))
print out
decision = out[0].argmax()
print(class_names[decision])
