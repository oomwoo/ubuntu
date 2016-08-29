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

import os
from neon.initializers import Uniform
from neon.layers import Affine, Conv, Pooling, GeneralizedCost
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Misclassification, Rectlin, Softmax, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.data.imageloader import ImageLoader
from neon.backends import gen_backend

# parse the command line arguments

home_dir = os.path.expanduser("~")
data_dir = home_dir + "/ubuntu/neon/"
# file_prefix = "trained/trained_bot_model_32c5516p22c3332p22a50e"
file_prefix = "model/trained_bot_model_debug"
num_epochs = 2  # args.epochs
img_size = 64
param_file_name = file_prefix + str(num_epochs) + ".prm"


be = gen_backend(backend='cpu', batch_size=128)

# Define CNN
train = ImageLoader(repo_dir=data_dir, set_name='train',
                        inner_size=img_size,
                        scale_range=0,  # Force scaling to match inner_size
                        do_transforms=False,
                        shuffle=True,
                        contrast_range=(75, 125))

test = ImageLoader(repo_dir=data_dir, set_name='validation',
                    inner_size=img_size,
                    do_transforms=False,
                    scale_range=0,  # Force scaling to match inner_size
                    shuffle=True,
                    contrast_range=(75, 125))


init_uni = Uniform(low=-0.1, high=0.1)
opt_gdm = GradientDescentMomentum(learning_rate=0.01,
                                  momentum_coef=0.9)

bn = True
layers = [Conv((5, 5, 16), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Pooling((2, 2)),
          Conv((3, 3, 32), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Pooling((2, 2)),
          Affine(nout=50, init=init_uni, activation=Rectlin(), batch_norm=bn),
          Affine(nout=4, init=init_uni, activation=Softmax())]

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

# Create model
mlp = Model(layers=layers)
callbacks = Callbacks(mlp, eval_set=test)  # Track cost function

# Train model
mlp.fit(train, optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

# Check performance
print 'Misclassification error = %.1f%%' % (mlp.eval(test, metric=Misclassification())*100)

# Save trained model
mlp.save_params(param_file_name)







##############
from PIL import Image
import numpy as np
from neon.data.dataiterator import ArrayIterator

W = img_size
H = img_size
L = W*H*3
size = H, W
class_names = ["forward", "left", "right", "backward"]    # from ROBOT-C bot.c
nclass = len(class_names)
x_new = np.zeros((128, L), dtype=np.float32)


test_file_name = home_dir + "/ubuntu/test/backward.jpg"
image = Image.open(test_file_name)
print("Loaded " + test_file_name)
image = image.resize(size)  # , Image.ANTIALIAS)
image = np.asarray(image, dtype=np.float32)
x_new[0] = image.reshape(1, L) / 255

test_file_name = home_dir + "/ubuntu/test/forward.jpg"
image = Image.open(test_file_name)
print("Loaded " + test_file_name)
image = image.resize(size)  # , Image.ANTIALIAS)
image = np.asarray(image, dtype=np.float32)
x_new[1] = image.reshape(1, L) / 255

test_file_name = home_dir + "/ubuntu/test/right.jpg"
image = Image.open(test_file_name)
print("Loaded " + test_file_name)
image = image.resize(size)  # , Image.ANTIALIAS)
image = np.asarray(image, dtype=np.float32)
x_new[2] = image.reshape(1, L) / 255

test_file_name = home_dir + "/ubuntu/test/left.jpg"
image = Image.open(test_file_name)
print("Loaded " + test_file_name)
image = image.resize(size)  # , Image.ANTIALIAS)
image = np.asarray(image, dtype=np.float32)
x_new[3] = image.reshape(1, L) / 255


# Run neural network
inference_set = ArrayIterator(x_new, None, nclass=nclass, lshape=(3, H, W))
out = mlp.get_outputs(inference_set)
print out

print(class_names[out[0].argmax()])
print(class_names[out[1].argmax()])
print(class_names[out[2].argmax()])
print(class_names[out[3].argmax()])
