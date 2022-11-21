import cv2
import numpy as np
import matplotlib.pyplot as plt
from .misc import *
from .io import *


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


def imshow(img, save_path=None):
  """
  Show an image with matplotlib.

  Parameters
  ----------
  img : np.ndarray, or list
    Image to show.
  save_path : str
    Path to save the figure.
  """
  if type(img) == np.ndarray:
    plt.imshow(img)
    if save_path is not None:
      plt.savefig(save_path)
    plt.show()
    plt.close()
  elif type(img) == list:
    imshow_grid(img)
  else:
    raise NotImplementedError('Unsupported type for visualization: ' + str(type(img)))


def imshow_grid(imgs, save_path=None):
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
  save_path : str
    Path to save the figure.
  """
  fig = plt.figure()
  n_rows = len(imgs)
  n_cols = len(imgs[0])
  for i, img in enumerate(sum(imgs, [])):
    fig.add_subplot(n_rows, n_cols, i+1)
    plt.imshow(img)
  if save_path is not None:
    plt.savefig(save_path)
  plt.show()
  plt.close(fig)


def render_dots_from_uv(uv, canvas, id_label=False, radius=None):
  if canvas.dtype != np.uint8:
    print('canvas must be uint8 type')
    exit(0)
  if radius is None:
    radius = int(max(round(canvas.shape[0] / 128), 1))
  font_scale = canvas.shape[0] / 480
  uv = np.round(uv).astype(np.int32)
  for i in range(uv.shape[0]):
    cv2.circle(canvas, (uv[i][1], uv[i][0]), radius, (255, 0, 0), thickness=-1)
    if id_label:
      cv2.putText(
        canvas, '%d' % i, (uv[i][1], uv[i][0]), cv2.FONT_HERSHEY_PLAIN,
        font_scale, (0, 0, 255)
      )
  return canvas


def render_bones_from_uv(uv, canvas, skeleton, valid=None, colors=None,
                         thickness=None, save_path=None):
  """
  Render bones from joint uv coordinates.

  Parameters
  ----------
  uv : np.ndarray, shape [k, 2]
    UV coordinates of joints.
  canvas : np.ndarray, dtype uint8
    Canvas to draw on.
  colors : list
    Colors for each bone. A list of list in range [0, 255].
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

  if colors is None:
    if hasattr(skeleton, 'colors'):
      colors = skeleton.colors
    else:
      colors = [[255, 0, 0]] * len(skeleton.parents)

  for child, parent in enumerate(skeleton.parents):
    if parent is None:
      continue
    else:
      c = colors[child]
    start = (int(uv[parent][1]), int(uv[parent][0]))
    end = (int(uv[child][1]), int(uv[child][0]))

    if valid is not None:
      if not valid[parent] or not valid[child]:
        continue

    cv2.line(canvas, start, end, c, thickness)

  if save_path is not None:
    save(save_path, canvas)

  return canvas


def render_bones_from_hmap(hmap, canvas, skeleton, valid=None, colors=None,
                           thickness=None, save_path=None):
  """
  Render bones from heat maps.

  Parameters
  ----------
  hmap : np.ndarray, shape [h, w, k]
    Heat maps.
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
  bones = render_bones_from_uv(
    coords, canvas, skeleton, valid, colors, thickness, save_path
  )
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
    plt.plot(xs, ys, zs, c=color_lib[p % len(color_lib)])
  plt.show()


def put_text(img, text, origin=None, color=(0, 255, 0), size=None):
  """
  origin : tuple, optional
    (x, y), by default None
  """
  font = cv2.FONT_HERSHEY_DUPLEX
  if size == None:
    size = int(math.ceil(max(img.shape) / 128))
  box = cv2.getTextSize(text, fontFace=font, fontScale=size, thickness=size)
  if origin is None:
    origin = (10, 10)
  origin = (origin[0], origin[1] + box[1])
  cv2.putText(img, text, origin, font, size, color=color, thickness=size)
  return img


def concat_videos(src_paths, tar_path, height=None, width=None):
  """
  You should only set height or width.
  """
  if height is None and width is None:
    raise RuntimeError('You must set either the height or the width.')
  readers = [VideoReader(p) for p in src_paths]
  writer = None
  for _ in progress_bar(readers[0].n_frames + 10):
    canvas = []
    for r in readers:
      frame = r.next_frame()
      if frame is None:
        canvas = []
        break
      canvas.append(imresize_diag(frame, width, height))
    if not canvas:
      break
    if height is None:
      canvas = np.concatenate(canvas, 0)
    else:
      canvas = np.concatenate(canvas, 1)
    if writer is None:
      writer = \
        VideoWriter(tar_path, canvas.shape[1], canvas.shape[0], readers[0].fps)
    writer.write_frame(canvas)
  for r in readers:
    r.close()
  writer.close()
