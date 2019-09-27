from core import *

def open3d_viewer_test():
  data = pkl_load('./test.pkl')
  verts = data['verts']
  faces = data['faces']
  render_sequence_3d(verts, faces, 1024, 512, './test_video.avi')


if __name__ == '__main__':
  open3d_viewer_test()
