import numpy as np
import cv2
import time
import uuid as uuid_import
import tqdm
import os
import matplotlib.pyplot as plt


color_lib = [
  [255, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25],
  [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
  [240, 50, 230], [210, 245, 60], [250, 190, 190], [0, 128, 128],
  [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
  [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128],
  [128, 128, 128]
]


def get_left_right_color(labels,
                         left=[255, 0, 0], right=[0, 255, 0], other=[0, 0, 255]):
  colors = []
  for l in labels:
    if 'left' in l:
      colors.append(left)
    elif 'right' in l:
      colors.append(right)
    else:
      colors.append(other)
  return colors


def imresize(img, size):
  """
  Resize an image.

  Parameters
  ----------
  img : np.ndarray
    Input image.
  size : tuple
    Size, (width, height).

  Returns
  -------
  np.ndarray
    Resized image.
  """
  return cv2.resize(img, size, cv2.INTER_LINEAR)


def imresize_diag(img, w=None, h=None):
  """
  Resize am image but keep the aspect ratio, according to target width or height.

  Parameters
  ----------
  img : [H, W, C]
    Image.
  w : int, optional
    Target width, by default None
  h : int, optional
    Target height, by default None

  Returns
  -------
  [H, W, C]
    Image.
  """
  if w is None:
    w = int(round(h / img.shape[0] * img.shape[1]))
  else:
    h = int(round(w / img.shape[1] * img.shape[0]))
  return imresize(img, (w, h))


def render_gaussian_hmap(centers, shape, sigma=None, dtype=np.float32):
  """
  Render gaussian heat maps from given centers.

  Parameters
  ----------
  centers : np.ndarray, shape [N, 2]
    Gussian centers, (row, column).
  shape : tuple
    Heatmap shape, (height, width)
  sigma : float, optional
    sigma, if None, would be height / 40

  Returns
  -------
  np.ndarray, float32, shape [height, width, N]
    Rendered heat maps
  """
  if sigma is None:
    sigma = shape[0] / 40
  x = [i for i in range(shape[1])]
  y = [i for i in range(shape[0])]
  xx, yy = np.meshgrid(x, y)
  xx = np.reshape(xx.astype(dtype), [shape[0], shape[1], 1])
  yy = np.reshape(yy.astype(dtype), [shape[0], shape[1], 1])
  x = np.reshape(centers[:,1], [1, 1, -1])
  y = np.reshape(centers[:,0], [1, 1, -1])
  distance = np.square(xx - x, dtype=dtype) + np.square(yy - y, dtype=dtype)
  hmap = np.exp(-distance / (2 * sigma**2 ), dtype=dtype) / \
         np.sqrt(2 * np.pi * sigma**2, dtype=dtype)
  hmap /= (np.max(hmap, axis=(0, 1), keepdims=True) + np.finfo(dtype).eps)
  return hmap


def print_important(s, sep='='):
  """
  Print important message.

  Parameters
  ----------
  s : str
    Message.
  sep : str, optional
    Seperator to fill one line, by default '='
  """
  print(sep*89)
  print(s)
  print(sep*89)


def arr_identical(a, b, verbose=True):
  """
  Wether two arrays are identical.

  Parameters
  ----------
  a : np.ndarray
    Array a.
  b : np.ndarray
    Array b.
  verbose : bool, optional
    Wether print detail infomation, by default True

  Returns
  -------
  bool
    Identical or not.
  """
  if a.shape != b.shape:
    if verbose:
      print('Different shape: a: {}, b: {}'.format(a.shape, b.shape))
    return False
  else:
    return np.allclose(a, b)


class Timer:
  """
  Timer to evaluate passed time.

  Call `tic` to save current time.

  """
  def __init__(self):
    self.last_tic = time.time()
    self.memory = {}

  def tic(self, key=None, reset=True):
    """
    Save current timestamp and return the interval from previous tic.

    Parameters
    ----------
    key : str, optional
      Key to save this interval, by default None
    reset : bool, optional
      If to reset the last tic time, by default True

    Returns
    -------
    float
      Interval time in second.
    """
    curr_time = time.time()
    interval = 0.0
    if self.last_tic is not None:
      interval = curr_time - self.last_tic
      if key is not None:
        self.memory[key] = interval
    if reset:
      self.last_tic = curr_time
    return interval


class LowPassFilter:
  """
  Lowpass filter. s = a * s + (1 - a) * s'

  Call 'process` to filter one signal.

  """
  def __init__(self, alpha=0.9):
    self.prev_raw_value = None
    self.prev_filtered_value = None
    self.alpha = alpha

  def process(self, value, alpha=None):
    """
    Filter the value.

    Parameters
    ----------
    value : np.ndarray
      Value to be filtered.

    Returns
    -------
    np.ndarray
      Filtered value.
    """
    if alpha is None:
      alpha = self.alpha
    if self.prev_raw_value is None:
      s = value
    else:
      s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
    self.prev_raw_value = value
    self.prev_filtered_value = s
    return s


class OneEuroFilter:
  """
  One euro filter.

  Call `process` to process the signal.

  Parameters
  ----------
  mincutoff : float, optional
    Decrease mincutoff to decrease slow speed jittering, by default 1.0
  beta : float, optional
    Increase beta to decrease speed lag, by default 0.0
  dcutoff : float, optional
    by default 1.0
  freq : int, optional
    by default 30
  """
  def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
    """
    Parameters
    ----------
    mincutoff : float, optional
      Decrease mincutoff to decrease slow speed jittering, by default 1.0
    beta : float, optional
      Increase beta to decrease speed lag, by default 0.0
    dcutoff : float, optional
      by default 1.0
    freq : int, optional
      by default 30
    """
    self.freq = freq
    self.mincutoff = mincutoff
    self.beta = beta
    self.dcutoff = dcutoff
    self.x_filter = LowPassFilter()
    self.dx_filter = LowPassFilter()

  def compute_alpha(self, cutoff):
    te = 1.0 / self.freq
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / te)

  def process(self, x):
    prev_x = self.x_filter.prev_raw_value
    dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
    edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
    cutoff = self.mincutoff + self.beta * np.abs(edx)
    return self.x_filter.process(x, self.compute_alpha(cutoff))


def hmap_to_uv(hmap):
  """
  Compute uv coordinates of heat map maxima in each layer.

  Parameters
  ----------
  hmap : np.ndarray, shape [h, w, k]
    Heat maps.

  Returns
  -------
  np.ndarray, shape [k, 2]
    UV coordinates in the order of (row, column).
  """
  shape = hmap.shape
  hmap = np.reshape(hmap, [-1, shape[-1]])
  v, h = np.unravel_index(np.argmax(hmap, 0), shape[:-1])
  coord = np.stack([v, h], 1)
  return coord


def camera_proj(k, xyz):
  """
  Camera projection: from camera space to image space.

  Parameters
  ----------
  k : np.ndarray, shape [3, 3]
    Camera intrinsics.
  xyz : np.ndarray, shape [n, 3]
    Coordinates.

  Returns
  -------
  np.ndarray, shape [n, 2]
    UV coordinates, (row, column)
  """
  uvd = np.dot(k, xyz.T)
  uv = np.flip((uvd / uvd[2:3, :])[:2].T, -1).copy()
  return uv


def camera_intrinsic(fx, fy, tx, ty):
  """
  Convert camera parameters into camera intrinsic matrix.

  Parameters
  ----------
  fx : float
    fx.
  fy : float
    fy
  tx : float
    tx
  ty : float
    ty

  Returns
  -------
  np.ndarray, shape [3, 3]
    Camear intrinsic matrix (placed on left).
  """
  return np.array([[fx, 0, tx], [0, fy, ty], [0, 0, 1]])


def compute_auc(xs, ys):
  """
  Compute area under curve (AuC).

  Parameters
  ----------
  xs : list
    A list of x values.
  ys : list
    A list of y values corresponding to x.

  Returns
  -------
  float
    Area under curve.
  """
  length = xs[-1] - xs[0]
  area = 0
  for i in range(len(ys) - 1):
    area += (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]) / 2 / length
  return area


def compute_pck(errors, thres_range, n_step):
  """
  Compute percentage of correct keypoints (PCK) under a range of thresholds.

  Parameters
  ----------
  errors : np.ndarray, shape [n]
    Errors of all joints of all test samples.
  thres_range : tuple
    Threshold range, (min, max)
  n_step : int, optional
    Number of steps between the threshold range, by default 16

  Returns
  -------
  list
    Error thresholds.
  list
    PCK for each threshold.
  """
  xs = np.linspace(thres_range[0], thres_range[1], num=n_step)
  ys = []
  for x in xs:
    ys.append(np.sum(errors < x) / errors.shape[0])
  return xs, ys


def uuid():
  """
  Get a random string.

  Returns
  -------
  str
    Random string.
  """
  return uuid_import.uuid4()


def progress_bar(producer, text=None):
  if type(producer) == int:
    producer = range(producer)
  return tqdm.tqdm(list(producer), ascii=True, desc=text, dynamic_ncols=True)


pbar = progress_bar


def get_bbox(uv, scale, limit,
             rand_scale=False, rand_min=1.0, rand_max=2.0, square=False):
  """
  Get the bounding box of give uv coordinates.

  Parameters
  ----------
  uv : [N, 2]
    UV coordinates.
  scale : float
    Scale of the bounding box.
  limit : [2]
    The bottom-right of the bbox will be clipped to [0, limit]
  rand_scale : bool, optional
    Random scaling, by default False
  rand_min : float, optional
    Min random scale factor, by default 1.0
  rand_max : float, optional
    Max random scale factor, by default 2.0
  square : bool, optional
    Wether to make the bbox square, by default False
  """
  tl = np.min(uv, axis=0)
  br = np.max(uv, axis=0)
  center = (br + tl) / 2

  size = br - tl
  if square:
    size = np.max(size)
  if rand_scale:
    size *= np.random.uniform(rand_min, rand_max)
  else:
    size *= scale

  tl = np.round(np.clip(center - size / 2, 0, limit)).astype(np.int32)
  br = np.round(np.clip(center + size / 2, 0, limit)).astype(np.int32)

  return tl, br


def press_to_continue(exit_0=True):
  """
  Wait for user input to continue. Enter 'n' to exit.

  Parameters
  ----------
  exit_0 : bool, optional
    If True, will directly exit(0), instead of return False, by default True

  Returns
  -------
  bool
    Continue or not.
  """
  if input('Continue? (enter n to exit) ') == 'n':
    if exit_0:
      exit(0)
    else:
      return False
  return True


wait = press_to_continue


def basic_statistics(data, axis=None, print_out=True):
  metrics = ['min', 'mean', 'max', 'std']
  info = [getattr(np, m)(data, axis) for m in metrics]
  s = ' '.join([f'{m} = {x:.2e}' for m, x in zip(metrics, info)])
  if print_out:
    print(s)
  return s


def inspect(data, indent=0, max_len=10):
  space = '  ' * indent
  print(space + f'Data type: {type(data)}')
  if type(data) == list:
    print(space + f'length: {len(data)}')
    print(space + 'first item:')
    inspect(data[0], indent=indent + 2)
  elif type(data) == str:
    print(space + f'String of length {len(data)}: {data}')
  elif type(data) == dict:
    print(space + f'Total items: {len(data)}')
    if len(data) > max_len:
      print(space + f'First {max_len} items:')
    cnt = 0
    for k, v in data.items():
      print(space, k, type(v))
      inspect(v, indent=indent + 2)
      cnt += 1
      if cnt >= max_len:
        break
  else:
    if hasattr(data, 'shape'):
      print(space + f'shape = {data.shape}')
    if hasattr(data, 'dtype'):
      print(space + f'dtype = {data.dtype}')
    if type(data) == np.ndarray:
      print(space + basic_statistics(data, axis=None, print_out=False))


examine_dict = inspect


def set_extension(file_name, ext):
  return os.path.splitext(file_name)[0] + '.' + ext


def count_model_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def hist(data, figsize=(12, 8), xlabel='', ylabel='', title='', save_path=None, show=False):
  plt.figure(figsize=figsize)
  _, _, patches = plt.hist(data, bins=20)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

  if title != '':
    title += '\n'
  title += f'mean {np.mean(data):.2f} std {np.std(data):.2f} '
  plt.title(title)

  for p in patches:
    plt.text(
      p.xy[0], p.xy[1] + p.get_height(),
      f'{p.xy[0]:.1f} - {p.xy[0] + p.get_width():.1f} \n {p.xy[1] + p.get_height() / len(data)*100:.2f}%'
    )

  if save_path is not None:
    plt.savefig(save_path)

  if show:
    plt.show()
