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


import subprocess
import os
import sys

def write_build_info():
    module_dir = os.path.dirname(os.path.realpath(__file__))
    try:
        f = open(os.path.join(module_dir, "nmt_chainer/_build.py"), "w")
    except OSError:
        print >>sys.stderr, "Warning: Could not create _build.py file"
        return
    
    try:
        git_hash =  subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                       cwd = module_dir,
                                       stderr = subprocess.STDOUT).strip()
    except:
        git_hash = "**Could not retrieve git-hash"
        
        
    try:
        DEVNULL = open(os.devnull, 'wb')
        returncode =  subprocess.call(['git', 'diff-index', '--quiet', 'HEAD', '--'], 
                                      cwd = module_dir, 
                                      stdout = DEVNULL,
                                      stderr = DEVNULL)
        if returncode == 0:
            git_dirty_status = "clean"
        elif returncode == 1:
            git_dirty_status = "dirty"
        else:
            git_dirty_status = "unknown"
    except:
        git_dirty_status = "unknown"
        
    try:
        git_diff = subprocess.check_output(['git', 'diff'], 
                                           cwd = module_dir,
                                            stderr = subprocess.STDOUT)
    except:
        git_diff =  '**Could not retrieve git-diff**'
        
    import json
    git_diff = json.dumps(git_diff)
        
    content="""# file created by setup.py
__build__ = "%s"

__dirty_status__ = "%s"
__git_diff__ = \"\"\"
%s
\"\"\"
"""%(git_hash, git_dirty_status, git_diff)
        
    f.write(content)
    

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

# create _build.py file
write_build_info()

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
      
      install_requires=['numpy>=1.10.0', 'chainer>=1.18.0', 'bokeh', 'plotly']
     )


