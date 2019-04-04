#!/bin/bash
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


if [[ -d preprocessed ]] && [[ -d raw ]]; then
    echo "data exist"
    exit 0
else
    wget -c http://dureader.gz.bcebos.com/demo.zip
    wget -c https://aipedataset.cdn.bcebos.com/dureader/dureader_raw.zip
    wget -c https://aipedataset.cdn.bcebos.com/dureader/dureader_preprocessed.zip
fi

if md5sum --status -c md5sum.txt; then
    unzip demo.zip
    unzip dureader_raw.zip
    unzip dureader_preprocessed.zip
else
    echo "download data error!" >> /dev/stderr
    exit 1
fi
