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


def render_bones_from_uv(uv, canvas, parents, color=None, thickness=None):
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
  color : list
    Color for each bone. A list of list in range [0, 255].
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
  for child, parent in enumerate(parents):
    if parent is None:
      continue
    if color is None:
      c = color_lib[child % len(color_lib)]
    else:
      c = color[child]
    start = (int(uv[parent][1]), int(uv[parent][0]))
    end = (int(uv[child][1]), int(uv[child][0]))

    anyzero = lambda x: x[0] * x[1] == 0
    if anyzero(start) or anyzero(end):
      continue

    cv2.line(canvas, start, end, c, thickness)
  return canvas


def render_bones_from_hmap(hmap, canvas, parents, color=None, thickness=None):
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
  bones = render_bones_from_uv(coords, canvas, parents, color, thickness)
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


def select_frames_from_video(video_path, save_prefix=None, fps=60, scale=1):
  """
  Read video, display frame by frame, and save selected frames.

  w: previous frame
  s: next frame
  a: revert
  d: forward
  q: quit
  space: save this frame

  Parameters
  ----------
  video_path : str
    Path to the video.
  save_prefix : str, optional
    Path prefix to save the frames, if None, will be the same as video_path,
    by default None
  fps : int, optional
    Display framerate, by default 60
  scale : float
    Display scale of the video frames, by default 1
  """
  import pygame

  if save_prefix is None:
    save_prefix = video_path

  reader = VideoReader(video_path)
  frames = reader.all_frames()
  reader.close()

  display_size = \
    (int(frames[0].shape[1] * scale), int(frames[0].shape[0] * scale))

  pygame.init()
  display = pygame.display.set_mode(display_size)
  pygame.display.set_caption('loading')

  idx = 0
  done = False
  clock = pygame.time.Clock()
  while not done:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        done = True
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_w:
          idx -= 1
        elif event.key == pygame.K_s:
          idx += 1
        elif event.key == pygame.K_SPACE:
          imsave('./frame_%d' % idx + '.png')
        elif event.key == pygame.K_q:
          done = True
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_a]:
      idx -= 1
    if pressed[pygame.K_d]:
      idx += 1

    idx = min(max(idx, 0), len(frames) - 1)
    pygame.display.set_caption('%s %d/%d' % (video_path, idx, len(frames)))
    display.blit(
      pygame.surfarray.make_surface(
        imresize(frames[idx], display_size).transpose((1, 0, 2))
      ), (0, 0)
    )
    pygame.display.update()

    clock.tick(fps)
