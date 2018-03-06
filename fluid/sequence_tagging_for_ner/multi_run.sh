#!/bin/bash

for i in 1 2 3 4 5 6 7 8 9 10
do
    echo $i
    python train.py >logfile_wending_$i 2>&1 &
done
