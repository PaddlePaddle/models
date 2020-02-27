from setuptools import setup
from setuptools import Extension

setup(
    name='pts_utils',
    ext_modules = [Extension(
        name='pts_utils',
        sources=['pts_utils.cpp'],
        include_dirs=[r'../../pybind11/include'],
        extra_compile_args=['-std=c++11']
    )],
)
