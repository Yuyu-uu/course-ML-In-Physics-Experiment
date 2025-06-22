import cv2
import matplotlib.pyplot as plt
import numpy as np
from Mask import create_mask
import tensorflow as tf
from PIL import Image
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from grab import main
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from DNNModel import D2NNOutput as Output
# from FashionMNISTPreprocess import dataset
from Mask import create_mask

@tf.function
def func_one_hot(ms):
    return tf.vectorized_map(fn=one_hot, elems=ms, fallback_to_while_loop=True)


def one_hot(m):
    ans0 = tf.math.reduce_sum(tf.math.multiply(m, masks[0]))
    ans1 = tf.math.reduce_sum(tf.math.multiply(m, masks[1]))
    ans2 = tf.math.reduce_sum(tf.math.multiply(m, masks[2]))
    ans3 = tf.math.reduce_sum(tf.math.multiply(m, masks[3]))
    ans4 = tf.math.reduce_sum(tf.math.multiply(m, masks[4]))
    ans5 = tf.math.reduce_sum(tf.math.multiply(m, masks[5]))
    ans6 = tf.math.reduce_sum(tf.math.multiply(m, masks[6]))
    ans7 = tf.math.reduce_sum(tf.math.multiply(m, masks[7]))
    ans8 = tf.math.reduce_sum(tf.math.multiply(m, masks[8]))
    ans9 = tf.math.reduce_sum(tf.math.multiply(m, masks[9]))
    answers = tf.concat([[ans0], [ans1], [ans2], [ans3], [ans4],
                         [ans5], [ans6], [ans7], [ans8], [ans9]], axis=0)
    return answers


dimension = 200  # Number of pixels on each side of input plane
phase_dimension = 600  # Number of pixels on each side of phase modulation layer
lmb = 532e-9  # Wavelength (um)
MSE_part = 0.8  # loss = MSE loss * MSE_part + SCE loss * (1 - MSE_part)
SCE_part = 0.2
stddev = 0.  # phase errors
phase_init = 0.
mask_shape = '343'  # Mask shape (343, 3223, circle)
lr_base = 0.1
lr_decay = 0.99
epochs = 10
batch_size = 128
buffer_size = 600
total_batch = 60000 // batch_size

is_view = True  # If output intensity is needed
view_every = 1000
is_test = True
is_save = True
border = 0  # 0s around the input digit
mask_border = 50  # 0s around the output plane
mask_on = True  # If the optical field is covered with the mask
normalize = True
rang_size_i = 30
rang_size = np.ones((10,)) * rang_size_i
base = 512
d1 = 218e-3  # mm
# d2 = 0e3  # mm
d2 = 177e-3
# d3 = 0.05
ds = [d1, d2]
amp_or_phase = ['phase']

masks, mask_location, rect_location = create_mask(shape=mask_shape, dim=dimension,
                                                      border=mask_border, rang_size=rang_size)

image_dir = r"/Users/yuyu/Documents/3xia/ML_Physics/光神经网络/DNN/ShowD2NN/imgs/24.png"
phase_dir = r"/Users/yuyu/Documents/3xia/ML_Physics/光神经网络/DNN/ShowD2NN/phase600/phase.npy"

phase_input = np.load(phase_dir)
img = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)

model_out = Output(layers=1, dim_in=dimension, dim_out=dimension, phase_dim=phase_dimension, lmb=lmb, ds=ds,
                       amp_or_phase=amp_or_phase)
out = model_out(input_image=img, phase_input=phase_input)

"""cx = 1015
cy = 993
ccdpx = 5.5e-6
SLMpx = 6.4e-6
factor = ccdpx/SLMpx
# 200 coord / fact = img coord
width = 200/factor
xstart = int(cx-width/2)
xend = int(cx+width/2-1)
ystart = int(cy-width/2)
yend = int(cy+width/2-1)

img = oimg[ystart:yend,xstart:xend]
img = img / np.max(img) * 255

out = np.flip(img,axis=1)
out = cv2.resize(out, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)"""

fig = plt.figure(figsize=(12, 8))
ans = tf.math.argmax(one_hot(out), axis=-1).numpy()
ax = fig.add_subplot(1, 2, 1)
divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig15 = ax.get_figure()
fig15.add_axes(ax_cb)
ax.set_title("Predicted: %d" % (ans))
if mask_border == 0:
    im = ax.imshow(out, cmap="gray")
else:
    im = ax.imshow((out)[mask_border:-mask_border, mask_border:-mask_border], cmap="gray")
ax.axis('off')
plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()
edge = rang_size * (dimension - mask_border * 2) // base
edge_color = "red"
for number in range(10):
    rect = Rectangle(rect_location[number], edge[number], edge[number], edgecolor=edge_color,
                        fill=False, linestyle='--', linewidth=1)
    ax.add_patch(rect)
ax.axis('off')
ax = fig.add_subplot(1, 2, 2)
ax.set_title("Phase 0")
divider = make_axes_locatable(ax)
ax_cb = divider.new_horizontal(size="5%", pad=0.05)
fig6 = ax.get_figure()
fig6.add_axes(ax_cb)
im = ax.imshow(img, cmap="gray")
plt.colorbar(im, cax=ax_cb)
ax_cb.yaxis.tick_right()
ax.axis('off')
plt.show()