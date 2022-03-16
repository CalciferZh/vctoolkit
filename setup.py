import setuptools

setuptools.setup(
  name='vctoolkit',
  version='0.1.9',
  author_email='yuxiao.zhou@outlook.com',
  description='A simple wrapper for commonly used tools in visual computing.',
  url='https://github.com/CalciferZh/vctoolkit',
  packages=setuptools.find_packages(),
  classifiers=["Programming Language :: Python :: 3"],
  install_requires=[
    'opencv-python', 'h5py', 'imageio', 'numpy', 'scipy',
    'transforms3d', 'tqdm', 'matplotlib'
  ]
)
