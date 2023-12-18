from setuptools import setup, find_packages

setup(
  name = 'xiadpm',
  packages = find_packages(exclude=[]),
  include_package_data=True,
  entry_points={
    'console_scripts': [],
  },
  version='0.0.0',
  license='MIT',
  description='PyTorch implementation of (Xia et Al., 2019)',
  author='Lemuel Puglisi',
  author_email = 'lemuel.puglisi@phd.unict.it',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'medical imaging',
    'disease progression modeling',
    'alzheimer'
  ],
  install_requires=[],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
  ],
)