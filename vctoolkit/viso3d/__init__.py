import open3d as o3d
import numpy as np
from ..io import VideoWriter
from tqdm import tqdm


def render_sequence_3d(verts, faces, width, height, video_path, fps=30,
                       visible=False, need_norm=True):
  """
  Render mesh animation using open3d.

  Parameters
  ----------
  verts : np.ndarray, shape [n, v, 3]
    Mesh vertices for each frame.
  faces : np.ndarray, shape [f, 3]
    Mesh faces.
  width : int
    Width of video.
  height : int
    Height of video.
  video_path : str
    Path to save the rendered video.
  fps : int, optional
    Video framerate, by default 30
  visible : bool, optional
    Wether to display rendering window, by default False
  need_norm : bool, optional
    Normalizing the vertices and locate camera automatically or not, by default
    True
  """
  if need_norm:
    if type(verts) == list:
      verts = np.stack(verts, 0)

    mean = np.mean(verts, axis=(0, 1), keepdims=True)
    scale = np.max(np.std(verts, axis=(0, 1), keepdims=True)) * 6
    verts = (verts - mean) / scale

  cam_offset = 1.2

  writer = VideoWriter(video_path, width, height, fps)

  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(faces)
  mesh.vertices = o3d.utility.Vector3dVector(verts[0])

  vis = o3d.visualization.Visualizer()
  vis.create_window(width=width, height=height, visible=visible)
  vis.add_geometry(mesh)
  view_control = vis.get_view_control()
  cam_params = view_control.convert_to_pinhole_camera_parameters()
  cam_params.extrinsic = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, cam_offset],
    [0, 0, 0, 1],
  ])
  view_control.convert_from_pinhole_camera_parameters(cam_params)

  for v in tqdm(verts, ascii=True):
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.compute_vertex_normals()
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    frame = (np.asarray(vis.capture_screen_float_buffer()) * 255).astype(np.uint8)
    writer.write_frame(frame)

  writer.close()


def create_o3d_mesh(verts, faces):
  """
  Create a open3d mesh from vertices and faces.

  Parameters
  ----------
  verts : np.ndarray, shape [v, 3]
    Mesh vertices.
  faces : np.ndarray, shape [f, 3]
    Mesh faces.

  Returns
  -------
  o3d.geometry.TriangleMesh
    Open3d mesh.
  """
  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(faces)
  mesh.vertices = o3d.utility.Vector3dVector(verts)
  mesh.compute_vertex_normals()
  return mesh


def vis_mesh(verts, faces, width=1080, height=1080):
  """
  Visualize mesh with open3d.

  Parameters
  ----------
  verts : np.ndarray, shape [v, 3]
    Mesh vertices.
  faces : np.ndarray, shape [f, 3]
    Mesh faces.
  width : int
    Window width, by default 1080.
  height : int
    Window height, by default 1080.
  """
  mesh = create_o3d_mesh(verts, faces)
  viewer = o3d.visualization.Visualizer()
  viewer.create_window(width=width, height=height, visible=True)
  viewer.add_geometry(mesh)
  viewer.run()

