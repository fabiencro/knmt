#!/usr/bin/env python

from distutils.core import setup

setup(name='nmt_chainer',
      version='1.0',
      description='Implementation of RNNSearch and other Neural MT models in Chainer',
      author='Fabien Cromieres',
      author_email='fabien.cromieres@gmail.com',
      url='https://github.com/fabiencro/knmt/',
      packages=['nmt_chainer'],
      license='GPL-3',
      install_requires=['chainer'],
     )

