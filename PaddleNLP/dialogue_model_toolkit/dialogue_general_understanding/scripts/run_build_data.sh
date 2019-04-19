#!/bin/bash

TASK_DATA=$1
typeset -l TASK_DATA

if [ "${TASK_DATA}" = "udc" ]
then
  exit 0
elif [ "${TASK_DATA}" = "swda" ]
then
  python build_swda_dataset.py
elif [ "${TASK_DATA}" = "mrda" ]
then
  python build_mrda_dataset.py
elif [[ "${TASK_DATA}" =~ "atis" ]]
then
  python build_atis_dataset.py
  cat ../data/atis/atis_slot/test.txt > ../data/atis/atis_slot/dev.txt
  cat ../data/atis/atis_intent/test.txt > ../data/atis/atis_intent/dev.txt
elif [ "${TASK_DATA}" = "dstc2" ]
then
  python build_dstc2_dataset.py
else
  echo "can not support $TASK_DATA , please choose [swda|mrda|atis|dstc2|multi-woz]"
fi
