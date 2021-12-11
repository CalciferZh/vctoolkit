import numpy as np
import cv2
import time
import uuid as uuid_import
import transforms3d
import tqdm
import os

from .io import load_hdf5


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
    self.last_tic = time.time()
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
  delta = np.zeros([len(parents), 3], np.float32)
  for c, p in enumerate(parents):
    if p is None:
      continue
    else:
      delta[c] = xyz[c] - xyz[p]
  length = np.linalg.norm(delta, axis=-1, keepdims=True)
  if norm_delta:
    delta /= np.maximum(length, np.finfo(np.float32).eps)
  return delta, length


def delta_to_xyz(delta, parents, length=None):
  """
  Convert bone orientations to joint coordinates.

  Parameters
  ----------
  delta : np.ndarray, shape [k, 3]
    Bone orientations.
  parents : list
    Parent for each joint.
  length : list or None
    The length of each bone (if delta are unit vectors), or None.

  Returns
  -------
  np.ndarray, [n, 3]
    Joint coordinates.
  """
  if length is None:
    length = np.ones(len(parents))

  xyz = [None] * len(parents)
  done = False
  while not done:
    done = True
    for c, p in enumerate(parents):
      if xyz[c] is not None:
        continue
      if p is None:
        xyz[c] = delta[c] * length[c]
      else:
        if xyz[p] is None:
          done = False
        else:
          xyz[c] = xyz[p] + delta[c] * length[c]
  xyz = np.stack(xyz, 0)
  return xyz


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


def axangle_to_rotmat(vec):
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


def progress_bar(producer, text=None):
  return tqdm.tqdm(list(producer), ascii=True, desc=text, dynamic_ncols=True)


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


class ParallelReader:
  def __init__(self, readers):
    """
    Always return the batch data of every reader concatenated along axis 0.

    Parameters
    ----------
    readers : list
      List of readers.
    """
    self.readers = readers
    self.batch_size = sum([r.batch_size for r in readers], 0)

  def next_batch(self):
    packs = [r.next_batch() for r in self.readers]
    batch_data = {}

    for k in packs[0].keys():
      batch_data[k] = np.concatenate([p[k] for p in packs], 0)
    return batch_data


class CascadeReader:
  def __init__(self, readers):
    """
    Always return the batch data of only one reader from the readers.

    Parameters
    ----------
    readers : list
      List of readers.
    """
    self.readers = readers
    self.batch_size = readers[0].batch_size
    self.reader_idx = 0

  def next_batch(self):
    if self.reader_idx == len(self.readers):
      self.reader_idx = 0
    data = self.readers[self.reader_idx].next_batch()
    self.reader_idx += 1
    return data


class DataLoader:
  def __init__(self, data_dir, batch_size):
    """
    Data loader to load all data from the data dir.

    Parameters
    ----------
    data_dir : str
      Data folder.
    batch_size : int
      Batch size.
    """
    print_important('Data loader at %s, batch size %d.' % (data_dir, batch_size))
    self.data_dir = data_dir
    self.data_files = os.listdir(self.data_dir)
    self.data_file_idx = 0
    self.cache_idx = 0
    self.batch_size = batch_size
    self.cache = {}
    self.cache_size = 0
    self._load_cache()

  def _load_cache(self):
    if self.data_file_idx == len(self.data_files):
      self.data_file_idx = 0
    if self.data_file_idx == 0:
      np.random.shuffle(self.data_files)
    self.cache_idx = 0

    load_success = False
    while not load_success:
      try:
        self.cache = load_hdf5(
          os.path.join(self.data_dir, self.data_files[self.data_file_idx])
        )
        load_success = True
      except Exception as e:
        print(e)
        print(self.data_dir)
        print(self.data_files[self.data_file_idx])
        time.sleep(10)
    self.data_file_idx += 1

    self.cache_size = 0
    for k, v in self.cache.items():
      if self.cache_size == 0:
        self.cache_size = v.shape[0]
      else:
        if self.cache_size != v.shape[0]:
          print('Error: all the data should have the same size along axis 0.')
          print('In %s - %s' % (self.data_files[self.data_file_idx - 1], k))
          print('%d (expected) vs %d (practical)' % (self.cache_size, v.shape[0]))
          exit(0)

    shuffled_indices = np.random.permutation(self.cache_size)

    for k, v in self.cache.items():
      self.cache[k] = v[shuffled_indices]

  def next_batch(self):
    start = self.cache_idx
    end = self.cache_idx + self.batch_size
    if end > self.cache_size:
      self._load_cache()
      return self.next_batch()
    self.cache_idx = end
    data = {k: v[start:end] for k, v in self.cache.items()}
    return data


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


def examine_dict(data):
  for k, v in data.items():
    if type(v) == np.ndarray:
      print(k, type(v), v.shape)
    else:
      print(k, type(v))


def set_extension(file_name, ext):
  return os.path.splitext(file_name)[0] + '.' + ext


def sphere_sampling(n):
  theta = np.random.uniform(0, 2 * np.pi, size=n)
  phi = np.random.uniform(0, np.pi, size=n)
  v = np.stack(
    [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1
  )
  return v


def random_rotation(n):
  axis = sphere_sampling(n)
  angle = np.random.uniform(np.pi, size=[n, 1])
  return convert(axis * angle, 'axangle', 'rotmat')


def slerp(a, b, t, batch=False):
  if not batch:
    a = np.expand_dims(a, 0)
    b = np.expand_dims(b, 0)
  dot = np.einsum('njd, njd -> nj', a, b)
  omega = np.expand_dims(np.arccos(np.clip(dot, 0, 1), dtype=np.float32), -1)
  so = np.sin(omega, dtype=np.float32)
  so[so == 0] = np.finfo(np.float32).eps
  p = np.sin((1 - t) * omega, dtype=np.float32) / so * a + \
      np.sin(t * omega, dtype=np.float32) / so * b
  mask = np.tile(np.prod(a == b, axis=-1, keepdims=True, dtype=np.bool), (1, 4))
  p = np.where(mask, a, p)
  if not batch:
    p = p[0]
  return p
