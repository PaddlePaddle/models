#!/bin/bash

cp -r configs/* PALM/config/
cp configs/mtl_config.yaml PALM/
rm -rf PALM/data
mv data PALM/
mv squad2_model PALM/pretrain_model
cp run_multi_task.sh PALM/
