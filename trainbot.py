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

#import numpy as np
from neon.initializers import Uniform
from neon.layers import Affine, Conv, Pooling, GeneralizedCost
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Misclassification, Rectlin, Softmax, CrossEntropyMulti
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from neon.data.imageloader import ImageLoader

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# hyperparameters
num_epochs = args.epochs

#(X_train, y_train), (X_test, y_test), nclass = load_cifar10(path=args.data_dir)

#train = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(3, 32, 32))
#test = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(3, 32, 32))

train = ImageLoader(repo_dir='/home/ilia/nervana/data', set_name='train',
                        inner_size=64,
                        scale_range=0,  # Force scaling to match inner_size
                        do_transforms=False,
                        shuffle=True,
                        contrast_range=(75,125))

test = ImageLoader(repo_dir='/home/ilia/nervana/data', set_name='validation',
                    inner_size=64,
                    do_transforms=False,
                    scale_range=0,  # Force scaling to match inner_size
                    shuffle=True,
                    contrast_range=(75, 125))


init_uni = Uniform(low=-0.1, high=0.1)
opt_gdm = GradientDescentMomentum(learning_rate=0.01,
                                  momentum_coef=0.9,
                                  stochastic_round=args.rounding)

bn = True
layers = [Conv((5, 5, 16), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Pooling((2, 2)),
          Conv((5, 5, 32), init=init_uni, activation=Rectlin(), batch_norm=bn),
          Pooling((2, 2)),
          Affine(nout=100, init=init_uni, activation=Rectlin(), batch_norm=bn),
          Affine(nout=4, init=init_uni, activation=Softmax())]

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, eval_set=test, **args.callback_args)

mlp.fit(train, optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

print 'Misclassification error = %.1f%%' % (mlp.eval(test, metric=Misclassification())*100)
