import time
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pixellib.semantic import semantic_segmentation
import tensorflow as tf
from MiDaS import depth_utils
import torch

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

segment_image = semantic_segmentation()
segment_image.load_ade20k_model("deeplabv3_xception65_ade20k.h5")

image_path = './images/DSC00005.JPG'

image = cv2.imread(image_path)[:, :, ::-1]
plt.title("Input Image")
plt.imshow(image), plt.show()
image = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(image)))

raw_labels, result = segment_image.segmentAsAde20k(image, is_array=True)

depth_utils.load_model()

tf.keras.backend.clear_session()
torch.cuda.empty_cache()
depth = depth_utils.predict_depth(image)

depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
art = np.where(result == (50, 50, 80), depth, image)
plt.title("Floor Depth Estimation")
plt.imshow(art), plt.show()
depth = depth[:, :, 0]

out = -75.629 + 254.129 * np.exp(-0.0011 * (255 - depth.astype('float32')))

points = []
height, width, depth = image.shape
for i in range(height):
    for j in range(width):
        if i % 5 == 0:
            if j % 5 == 0:
                if np.array_equal(result[i, j], (50, 50, 80)):
                    points.append([j, height - i, out[i, j]])

N_POINTS = len(points)
EXTENTS = 5
NOISE = 5

points = np.array(points)

xs = points[:, 0]
ys = points[:, 1]
zs = points[:, 2]

plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(xs, ys, zs, color='b')

tmp_A = []
tmp_b = []
for i in range(len(xs)):
    tmp_A.append([xs[i], ys[i], 1])
    tmp_b.append(zs[i])
b = np.matrix(tmp_b).T
A = np.matrix(tmp_A)

fit = (A.T * A).I * A.T * b
errors = b - A * fit
residual = np.linalg.norm(errors)

# print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))

xlim = ax.get_xlim()
ylim = ax.get_ylim()
X, Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                   np.arange(ylim[0], ylim[1]))
Z = np.zeros(X.shape)
for r in range(X.shape[0]):
    for c in range(X.shape[1]):
        Z[r, c] = fit[0] * X[r, c] + fit[1] * Y[r, c] + fit[2]
ax.plot_wireframe(X, Y, Z, color='k')
ax.plot_surface(X, Y, Z, color='r', alpha=0.4)

plt.title("Fitted Plane Equation: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim([0, 1000])
ax.set_ylim([0, 500])
ax.set_zlim([100, 350])
plt.show()
