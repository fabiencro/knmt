#!/usr/bin/env python

# from distutils.core import setup
from setuptools import setup, find_packages

setup(name='nmt_chainer',
      version='1.0',
      description='Implementation of RNNSearch and other Neural MT models in Chainer',
      author='Fabien Cromieres',
      author_email='fabien.cromieres@gmail.com',
        #packages=find_packages()
      packages=['nmt_chainer'],
      url="https://github/fabiencro/knmt",
      scripts=["nmt_chainer/knmt"],
      install_requires=['chainer', 'bokeh', 'plotly']
     )

