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
from neon.util.argparser import NeonArgparser
from neon.data.imageloader import ImageLoader

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

home_dir = os.path.expanduser("~")
data_dir = home_dir + "/nervana/data/"
file_prefix = "model/trained_bot_model_64c5516p22c5532p22a50e"
num_epochs = 30  # args.epochs
param_file_name = file_prefix + str(num_epochs) + ".prm"

# Define CNN
train = ImageLoader(repo_dir=data_dir, set_name='train',
                        inner_size=64,
                        scale_range=0,  # Force scaling to match inner_size
                        do_transforms=False,
                        shuffle=True,
                        contrast_range=(75,125))

test = ImageLoader(repo_dir=data_dir, set_name='validation',
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
          Affine(nout=50, init=init_uni, activation=Rectlin(), batch_norm=bn),
          Affine(nout=4, init=init_uni, activation=Softmax())]

cost = GeneralizedCost(costfunc=CrossEntropyMulti())

# Create model
mlp = Model(layers=layers)
callbacks = Callbacks(mlp, eval_set=test, **args.callback_args)  # Track cost function

# Train model
mlp.fit(train, optimizer=opt_gdm, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

# Check performance
print 'Misclassification error = %.1f%%' % (mlp.eval(test, metric=Misclassification())*100)

# Save trained model
mlp.save_params(param_file_name)

# dt = datetime.datetime.now()
# s = dt.strftime("%b_%d_%Y_%H_%M")
# mlp.save_params("trained_bot_model_" + s + ".prm")
