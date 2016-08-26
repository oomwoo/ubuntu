#!/usr/bin/env python
# Train robot to drive autonomously using Nervana Neon
# See more at https://github.com/oomwoo/
#
# Copyright (C) 2016 oomwoo.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License version 3.0
# as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License <http://www.gnu.org/licenses/> for details.
"""
Train robot to drive autonomously. Use a small convolutional neural network.
"""

# parse the command line arguments
from neon.util.argparser import NeonArgparser
parser = NeonArgparser(__doc__)
args = parser.parse_args()
args.batch_size = 1

# Set up backend for inference
from neon.backends import gen_backend
be = gen_backend(backend='cpu', batch_size=1)

from neon.layers import Affine, Conv, Pooling, GeneralizedCost
from neon.models import Model
from neon.transforms import Rectlin, Softmax
from neon.initializers import Uniform


param_file_name = "trained_bot_model.prm"

init_uni = Uniform(low=-0.1, high=0.1)

bn = True
layers = [Conv((5, 5, 16), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Pooling((2, 2)),
          Conv((5, 5, 32), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Pooling((2, 2)),
          Affine(nout=100, init=init_uni, activation=Rectlin(), batch_norm=bn),
          Affine(nout=4, init=init_uni, activation=Softmax())]

#model = Model(param_file_name)
model = Model(layers=layers)
model.load_params(param_file_name, load_states=False)

# crop and resize to 32x32
from PIL import Image
import numpy as np

img = Image.open('image.jpg')
crop = img.crop((0,0,min(img.size),min(img.size)))
crop.thumbnail((32, 32))
crop = np.asarray(crop, dtype=np.float32)

import numpy as np
x_new = np.zeros((128, 3072), dtype=np.float32)
x_new[0] = crop.reshape(1, 3072) / 255

inference_set = ArrayIterator(x_new, None, nclass=nclass, lshape=(3, 32, 32))

classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
out = model.get_outputs(inference_set)
print classes[out[0].argmax()]



# setup buffers
#xbuf = np.zeros((sentence_length, 1), dtype=np.int32)  # host buffer
#xdev = be.zeros((sentence_length, 1), dtype=np.int32)  # device buffer
#outputs = model.fprop(x, inference=True)
