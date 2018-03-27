#!/bin/bash

git clone https://github.com/chinese-poetry/chinese-poetry.git

if [ ! -d raw ]
then
    mkdir raw
fi

mv chinese-poetry/json/poet.tang.* raw/
rm -rf chinese-poetry
