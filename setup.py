from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

# Define all Cython extensions manually
cython_modules = [
    "irf.tree._utils",
    "irf.tree._criterion",
    "irf.tree._splitter",
    "irf.tree._tree"
]

extensions = [
    Extension(
        name=mod,
        sources=[mod.replace(".", "/") + ".pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    for mod in cython_modules
]

setup(
    name="irf",
    version="0.1",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"}
    ),
    zip_safe=False,
)