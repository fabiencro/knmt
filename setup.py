#!/usr/bin/env python

# from distutils.core import setup
from setuptools import setup, find_packages

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: Implementation :: CPython",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(name='knmt',
      version='0.9',
      description='Implementation of RNNSearch and other Neural MT models in Chainer',
      author='Fabien Cromieres',
      author_email='fabien.cromieres@gmail.com',
        #packages=find_packages()
      packages=['nmt_chainer'],
      url="https://github/fabiencro/knmt",
      scripts=["nmt_chainer/knmt"],
      install_requires=['chainer', 'bokeh', 'plotly']
     )

