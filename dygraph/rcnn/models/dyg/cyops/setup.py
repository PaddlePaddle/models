from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

import numpy as np

_NP_INCLUDE_DIRS = np.get_include()

# Extension modules
ext_modules = [
    Extension(
        name='cyops.bbox',
        sources=['cyops/bbox.pyx'],
        extra_compile_args=['-Wno-cpp'],
        include_dirs=[_NP_INCLUDE_DIRS]), Extension(
            name='cyops.nms',
            sources=['cyops/nms.pyx'],
            extra_compile_args=['-Wno-cpp'],
            include_dirs=[_NP_INCLUDE_DIRS]), Extension(
                name='cyops.target',
                sources=['cyops/target.pyx'],
                extra_compile_args=['-Wno-cpp'],
                include_dirs=[_NP_INCLUDE_DIRS])
]

setup(
    name='dyg_rcnn',
    #packages=['detectron'],
    language='c',
    ext_modules=cythonize(ext_modules))
