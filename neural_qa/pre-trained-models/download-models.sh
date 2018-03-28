#!/bin/bash
if [[ -f params_pass_00010.tar.gz ]] && [[ -f params_pass_00021.tar.gz ]]; then
  echo "data exist"
  exit 0
else
  wget -c http://cloud.dlnel.org/filepub/?uuid=d9a00599-1f66-4549-867b-e958f96474ca \
    -O neural_seq_qa.pre-trained-models.2017-10-27.tar.gz
fi

if [[ `md5sum -c neural_seq_qa.pre-trained-models.2017-10-27.tar.gz.md5` =~ 'OK' ]] ; then
  tar xf neural_seq_qa.pre-trained-models.2017-10-27.tar.gz
  rm neural_seq_qa.pre-trained-models.2017-10-27.tar.gz
else
  echo "download data error!" >> /dev/stderr
  exit 1
fi

