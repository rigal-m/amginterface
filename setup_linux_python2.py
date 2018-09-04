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

amgio = Extension(
    'amgio',
    ['src/amgiomodule.c', 'src/tools.c', 'src/amgio/amgio_py.c',
     'src/amgio/amgio_tools.c', 'src/amgio/convert.c', 'src/amgio/GMFio.c',
     'src/amgio/mesh.c', 'src/amgio/option.c', 'src/amgio/SU2io.c',
     'src/libmeshb/libmeshb7.c'],
    include_dirs=["src/", "src/amgio/", "src/libmeshb/", numpy.get_include()],
    extra_compile_args=["-DPYTHON_2", "-Di4"],
)


setup(
    name='amgio',
    version='0.0.0',
    url='',
    author='',
    author_email='',
    description='This module provides an interface for using the amg software',
    long_description=readme(),
    # packages=find_packages(),
    ext_modules=[amgio],
    packages=[''],
    package_dir={'': '.'},
    # include_package_data=True,
    # # package_data={
    # #     '': ['amgiomodule.so', './amgio/dep/*']
    # # },
    # distclass=BinaryDistribution,
    install_requires=[
        'numpy>=1.13.3',
    ],
    classifiers=[
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Topic :: Scientific/Engineering',
        'Development Status :: 1 - Planning',
    ],
)
