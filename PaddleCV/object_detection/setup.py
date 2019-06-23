#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Setup script.

Authors: dangqingqing(dangqingqing@baidu.com)
Date:    2019/04/22 14:21:11
"""

import setuptools
import sys

docstring_parser = 'docstring_parser @ ' \
    + 'http://github.com/willthefrog/docstring_parser/tarball/master'
install_requires = [docstring_parser]

if sys.version_info[0] > 2:
    install_requires += ['typeguard']

setuptools.setup(install_requires=install_requires, )
