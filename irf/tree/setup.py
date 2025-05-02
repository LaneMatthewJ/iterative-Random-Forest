import os
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

# Determine libraries to link
libraries = []
if os.name == 'posix':
    libraries.append('m')  # Link math library on POSIX systems

# List of Cython extension modules
extensions = [
    Extension(
        name="tree._tree",
        sources=["tree/_tree.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    ),
    Extension(
        name="tree._splitter",
        sources=["tree/_splitter.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    ),
    Extension(
        name="tree._criterion",
        sources=["tree/_criterion.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    ),
    Extension(
        name="tree._utils",
        sources=["tree/_utils.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="tree",
    version="0.1",
    packages=find_packages(where=".", include=["tree", "tree.*"]),
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"}
    ),
    zip_safe=False,
)