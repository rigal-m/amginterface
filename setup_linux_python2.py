import os
import numpy
from setuptools import setup, Extension, find_packages
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='amgio',
    version='0.0.0',
    url='',
    author='',
    author_email='',
    description='This module provides an interface for using the amg software',
    long_description=readme(),
    # packages=find_packages(),
    packages=[''],
    package_dir={'': '.'},
    include_package_data=True,
    package_data={
        '': ['amgiomodule.so', './amgio/dep/*']
    },
    distclass=BinaryDistribution,
    # install_requires=[
    #     'numpy>=1.13.3',
    # ],
    classifiers=[
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Topic :: Scientific/Engineering',
        'Development Status :: 1 - Planning',
    ],
)
