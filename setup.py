#!/usr/bin/env python

import os
from setuptools import setup
from setuptools.dist import Distribution

setup(
    name='nmt_chainer',
    version='1.0',
    description='Implementation of RNNSearch and other Neural MT models in Chainer',
    author='Fabien Cromieres',
    author_email='fabien.cromieres@gmail.com',
    url='https://github.com/fabiencro/knmt',
    packages=['nmt_chainer'],
    install_requires=[
        'bokeh>=0.12.2',
        'chainer==1.17.0', 
        'h5py>=2.6.0'
    ],
    entry_points={ 
        'console_scripts': [ 
            'make_data = nmt_chainer.make_data:cmdline',
            'train = nmt_chainer.train:command_line',
            'eval = nmt_chainer.eval:command_line'
        ] 
    }
)

