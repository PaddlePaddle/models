
cmake_minimum_required(VERSION 2.8.12)
project(pts_utils)

add_subdirectory(pybind11)
pybind11_add_module(pts_utils pts_utils.cpp)
