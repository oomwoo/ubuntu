from neon.backends import gen_backend
be = gen_backend(backend='cpu', batch_size=128)

from neon.data import load_cifar10
(X_train, y_train), (X_test, y_test), nclass = load_cifar10()

from neon.data import ArrayIterator
train_set = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(3, 32, 32))
test_set = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(3, 32, 32))

from neon.layers import Conv, Affine, Pooling
from neon.initializers import Uniform
from neon.transforms.activation import Rectlin, Softmax
init_uni = Uniform(low=-0.1, high=0.1)
layers = [Conv(fshape=(5,5,16), init=init_uni, activation=Rectlin()),
          Pooling(fshape=2, strides=2),
          Conv(fshape=(5,5,32), init=init_uni, activation=Rectlin()),
          Pooling(fshape=2, strides=2),
          Affine(nout=500, init=init_uni, activation=Rectlin()),
          Affine(nout=10, init=init_uni, activation=Softmax())]

from neon.models import Model
model = Model(layers)

from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyMulti
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

from neon.optimizers import GradientDescentMomentum, RMSProp
optimizer = GradientDescentMomentum(learning_rate=0.005,
                                    momentum_coef=0.9)

# Set up callbacks. By default sets up a progress bar
from neon.callbacks.callbacks import Callbacks
callbacks = Callbacks(model, train_set)

model.fit(dataset=train_set, cost=cost, optimizer=optimizer,  num_epochs=5, callbacks=callbacks)

model.save_params("cifar10_model.prm")

# Evaluate performance
from neon.transforms import Misclassification
error_pct = 100 * model.eval(test_set, metric=Misclassification())
print 'Misclassification error = %.1f%%' % error_pct

#  ###########################################
# an image of a frog from wikipedia
img_source = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Atelopus_zeteki1.jpg/440px-Atelopus_zeteki1.jpg"

# download the image
import urllib
urllib.urlretrieve(img_source, filename="image.jpg")

# crop and resize to 32x32
from PIL import Image
import numpy as np

img = Image.open('image.jpg')
crop = img.crop((0,0,min(img.size),min(img.size)))
crop.thumbnail((32, 32))
# crop.show()
crop = np.asarray(crop, dtype=np.float32)

# import numpy as np
x_new = np.zeros((128,3072), dtype=np.float32)
x_new[0] = crop.reshape(1,3072)/ 255

inference_set = ArrayIterator(x_new, None, nclass=nclass, lshape=(3, 32, 32))

classes =["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]
out = model.get_outputs(inference_set)
print classes[out[0].argmax()]
