#! /usr/bin/env python

import os
import sys
from os import path
import setuptools
import numpy
from numpy.distutils.core import setup
from setuptools import find_packages

path_to_repo = path.abspath(path.dirname(__file__))
with open(path.join(path_to_repo, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()
DISTNAME = 'irf'
DESCRIPTION = "irf"
URL = 'https://github.com/Yu-Group/irf'
MAINTAINER = 'Many'
MAINTAINER_EMAIL = 'many@berkeley.edu'
LICENSE = 'new BSD'
VERSION = '0.2.5'


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.add_subpackage('irf')

    return config


if __name__ == "__main__":
    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)
    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          packages=find_packages(),
          include_package_data=True,
          long_description=long_description,
          long_description_content_type="text/markdown",
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'
          ],
          install_requires=[
              "numpy<1.18",
              "scipy",
              "scikit-learn < 0.23",
              "cython<3",
              "pyfpgrowth",
              "pyspark",
              "pyyaml",
              "pydotplus",
              "matplotlib==3.5",
          ],
          setup_requires=["cython"])
