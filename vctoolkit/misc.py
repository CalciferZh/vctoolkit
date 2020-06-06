import numpy as np
import cv2
import time
import uuid as uuid_import
import transforms3d


color_lib = [
  [255, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25],
  [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
  [240, 50, 230], [210, 245, 60], [250, 190, 190], [0, 128, 128],
  [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
  [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128],
  [128, 128, 128]
]


def one_hot_decoding(array):
  """
  One-hot decoding.

  Parameters
  ----------
  array : np.ndarray
    Array encoded in the last dimension.

  Returns
  -------
  np.ndarray
    Array after decoding.
  """
  return np.argmax(array, axis=-1)


def one_hot_encoding(array, n_channels):
  """
  One-hot decoding.

  Parameters
  ----------
  array : np.ndarray
    Array to be encoded.
  n_channels : int
    Number of channels, i.e. number of categories.

  Returns
  -------
  np.ndarray
    Array after encoding.
  """
  shape = list(array.shape)
  array = np.reshape(array, [-1])
  array = np.eye(n_channels)[array]
  shape.append(n_channels)
  array = np.reshape(array, shape)
  return array


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


def render_gaussian_hmap(centers, shape, sigma=None):
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
  xx = np.reshape(xx.astype(np.float32), [shape[0], shape[1], 1])
  yy = np.reshape(yy.astype(np.float32), [shape[0], shape[1], 1])
  x = np.reshape(centers[:,1], [1, 1, -1])
  y = np.reshape(centers[:,0], [1, 1, -1])
  distance = np.square(xx - x) + np.square(yy - y)
  hmap = np.exp(-distance / (2 * sigma**2 )) / np.sqrt(2 * np.pi * sigma**2)
  hmap /= (
    np.max(hmap, axis=(0, 1), keepdims=True) + np.finfo(np.float32).eps
  )
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
    self.last_tic = 0
    self.memory = {}

  def tic(self, key=None):
    """
    Save current timestamp and return the interval from previous tic.

    Parameters
    ----------
    key : str, optional
      Key to save this interval, by default None

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


def xyz_to_delta(xyz, parents, norm_delta):
  """
  Convert joint coordinates to bone orientations (delta).

  Parameters
  ----------
  xyz : np.ndarray, shape [k, 3]
    Joint coordinates.
  parents : list
    Parent for each joint.
  norm_delta : bool
    Wether normalize bone deltas.

  Returns
  -------
  np.ndarray, [n, 3]
    Bone orientations.
  np.ndarray, [n, 1]
    Bone lengths.
  """
  delta = np.zeros([len(parents), 3])
  for c, p in enumerate(parents):
    if p is None:
      continue
    else:
      delta[c] = xyz[c] - xyz[p]
  length = np.linalg.norm(delta, axis=-1, keepdims=True)
  if norm_delta:
    delta /= np.maximum(length, np.finfo(np.float32).eps)
  return delta, length


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


def axangle_to_mat(vec):
  """
  Convert axis-angle rotation vector into rotation matrix.

  Parameters
  ----------
  vec : np.ndarray, shape [3]
    Rotation vector.

  Returns
  -------
  np.ndarray, shape [3, 3]
    Rotation matrix.
  """
  angle = np.linalg.norm(vec)
  axis = vec / angle
  return transforms3d.axangles.axangle2mat(axis, angle)
