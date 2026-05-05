from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    "_marching_cubes_lewiner_cy",
    sources=["_marching_cubes_lewiner_cy.pyx"],
    include_dirs=[numpy.get_include()],
)
setup(name="marching_cubes_cy", ext_modules=cythonize([ext]))
