# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件允许模块包以python -m ppdetection方式直接执行。

Authors: dangqingqing(dangqingqing@baidu.com)
Date:    2019/04/22 14:21:11
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import sys
from ppdetection.cmdline import main
sys.exit(main())
