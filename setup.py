#!/usr/bin/env python

# from distutils.core import setup
from setuptools import setup, find_packages

# getting the version string
# as suggested in http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
import re
VERSIONFILE="nmt_chainer/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# classifiers
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

# actual setup
setup(name='knmt',
      version = verstr,
      description='Implementation of RNNSearch and other Neural MT models in Chainer',
      author='Fabien Cromieres',
      author_email='fabien.cromieres@gmail.com',
        #packages=find_packages()
      packages=['nmt_chainer'],
      url="https://github.com/fabiencro/knmt",
#       scripts=["nmt_chainer/knmt.py"],

      scripts=["bin/knmt", "bin/knmt-debug", "bin/knmt-server"],
      
#       entry_points= {
#           'console_scripts': [
#               'knmt = nmt_chainer.__main__:main'
#           ]
#                      },
      
      install_requires=['chainer>=1.18.0', 'numpy>=1.10.0', 'bokeh', 'plotly']
     )


