#!/bin/bash

wget --no-check-certificate https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz
tar zxf dac.tar.gz
rm -f dac.tar.gz

mkdir raw
mv ./*.txt raw/
