import cv2
import numpy as np
import matplotlib.pyplot as plt
from .misc import *


def imshow_cv(img, caption='OpenCV Image Show'):
  """
  Show an image with opencv.

  Parameters
  ----------
  img : np.ndarray
    Input image in RGB format.
  caption : str, optional
    Window caption, by default 'OpenCV Image Show'
  """
  cv2.imshow(caption, np.flip(img, axis=-1).copy())
  cv2.waitKey()


def imshow(img):
  """
  Show an image with matplotlib.

  Parameters
  ----------
  img : np.ndarray
    Image to show.
  """
  plt.imshow(img)
  plt.show()


def imshow_grid(imgs, nrows, ncols):
  """
  Display multiple images as a grid.

  Parameters
  ----------
  imgs : list
    A list of images.
  nrows : int
    Number of rows.
  ncols : int
    Number of columns.
  """
  fig = plt.figure()
  for i, img in enumerate(imgs):
    fig.add_subplot(nrows, ncols, i+1)
    plt.imshow(img)
  plt.show()


def imshow_onerow(imgs):
  """
  Display images in one row.

  Parameters
  ----------
  imgs : list
    List of images to be displayed.
  """
  imshow_grid(imgs, 1, len(imgs))


def render_bones_from_uv(uv, canvas, parents, thickness=None):
  """
  Render bones from joint uv coordinates.

  Parameters
  ----------
  uv : np.ndarray, shape [k, 2]
    UV coordinates of joints.
  canvas : np.ndarray, dtype uint8
    Canvas to draw on.
  parents : list
    The parent joint for each joint. Root joint's parent should be None.
  thickness : int, optional
    Thickness of the line, by default None

  Returns
  -------
  np.ndarray
    The canvas after rendering.
  """
  if canvas.dtype != np.uint8:
    print('canvas must be uint8 type')
    exit(0)
  if thickness is None:
    thickness = int(max(round(canvas.shape[0] / 128), 1))
  for c, p in enumerate(parents):
    if p is None:
      continue
    color = color_lib[p]
    start = (int(uv[p][1]), int(uv[p][0]))
    end = (int(uv[c][1]), int(uv[c][0]))

    anyzero = lambda x: x[0] * x[1] == 0
    if anyzero(start) or anyzero(end):
      continue

    cv2.line(canvas, start, end, color, thickness)
  return canvas


def render_bones_from_hmap(hmap, canvas, parents, thickness=None):
  """
  Render bones from heat maps.

  Parameters
  ----------
  hmap : np.ndarray, shape [h, w, k]
    Heat maps.
  canvas : np.ndarray, shape [h, w, 3], dtype uint8
    Canvas to render.
  parents : list
    Parent joint of each child joint.
  thickness : int, optional
    Thickness of each bone, by default None

  Returns
  -------
  np.ndarray
    Canvas after rendering.
  """
  coords = hmap_to_uv(hmap)
  bones = render_bones_from_uv(coords, canvas, parents, thickness)
  return bones


def render_bones_plt(joints, parents):
  """
  Render bones in 3D with matplotlib.

  Parameters
  ----------
  joints : np.ndarray
    Joint positions.
  parents : list
    Parent joint of each joint.
  """
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = Axes3D(fig)

  ax.set_xlim3d(-1.5, 1.5)
  ax.set_ylim3d(-1.5, 1.5)
  ax.set_zlim3d(-1.5, 1.5)

  ax.grid(False)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])
  ax.set_axis_off()
  ax.view_init(-90, -90)

  for c, p in enumerate(parents):
    if p is None:
      continue
    xs = [joints[c, 0], joints[p, 0]]
    ys = [joints[c, 1], joints[p, 1]]
    zs = [joints[c, 2], joints[p, 2]]
    plt.plot(xs, ys, zs, c=color_lib[p])
  plt.show()
