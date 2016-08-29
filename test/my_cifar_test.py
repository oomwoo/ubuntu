from neon.backends import gen_backend
be = gen_backend(backend='cpu', batch_size=1)

from neon.data import ArrayIterator
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
model.load_params("cifar10_model.prm", load_states=False)

classes =["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]
nclass = len(classes)

#  ###########################################
# an image of a frog from wikipedia
# img_source = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Atelopus_zeteki1.jpg/440px-Atelopus_zeteki1.jpg"

# download the image
# import urllib
# urllib.urlretrieve(img_source, filename="image.jpg")

# crop and resize to 32x32
from PIL import Image
import numpy as np

img = Image.open('frog.jpg')
crop = img.crop((0,0,min(img.size),min(img.size)))
crop.thumbnail((32, 32))
# crop.show()
crop = np.asarray(crop, dtype=np.float32)

# import numpy as np
# x_new = np.zeros((128,3072), dtype=np.float32)
x_new = crop.reshape(1,3072)/ 255
inference_set = ArrayIterator(x_new, None, nclass=nclass, lshape=(3, 32, 32))
out = model.get_outputs(inference_set)
print classes[out[0].argmax()]

# ######
from neon.data import load_cifar10
(X_train, y_train), (X_test, y_test), nclass = load_cifar10()
i = 1
x_new = X_test[i]
x_new = x_new.reshape(3, 32, 32)
img = np.transpose(x_new, (1, 2, 0))
img = Image.fromarray(np.uint8(img*255))
img.show()

x_new = x_new.reshape(1, 3072)
test_set = ArrayIterator(x_new, None, nclass=nclass, lshape=(3, 32, 32))
out = model.get_outputs(test_set)
print classes[out.argmax()] + ", ground truth " + classes[y_test[i][0]]
