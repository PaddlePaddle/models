#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
# ======================================================================
"""this file is only for PaddleCloud"""

import os

import logging

import paddle.fluid.contrib.utils.hdfs_utils as hdfs_utils

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cloud")
logger.setLevel(logging.INFO)


def run():
    cmd = "python -u train.py "

    cmd += " --train_data_path %s " % "data/train.txt"

    cmd += " --test_data_path %s " % "data/test.txt"

    if os.getenv("BATCH_SIZE", ""):
        cmd += " --batch_size %s " % os.getenv("BATCH_SIZE")

    if os.getenv("EMBEDDING_SIZE", ""):
        cmd += " --embedding_size %s " % os.getenv("EMBEDDING_SIZE")

    if os.getenv("NUM_PASSES", ""):
        cmd += " --num_passes %s " % os.getenv("NUM_PASSES")

    if os.getenv("MODEL_OUTPUT_DIR", ""):
        cmd += " --model_output_dir %s " % os.getenv("MODEL_OUTPUT_DIR")

    if os.getenv("SPARSE_FEATURE_DIM", ""):
        cmd += " --sparse_feature_dim %s " % os.getenv("SPARSE_FEATURE_DIM")

    if os.getenv("ASYNC_MODE", ""):
        cmd += " --async_mode "

    if os.getenv("NO_SPLIT_VAR", ""):
        cmd += " --no_split_var "

    is_local = int(os.getenv("PADDLE_IS_LOCAL", "1"))

    if is_local:
        cmd += " --is_local 1 "
        cmd += " --cloud_train 0 "
    else:
        cmd += " --is_local 0 "
        cmd += " --cloud_train 1 "

        trainer_id = int(os.environ["PADDLE_TRAINER_ID"])
        trainers = int(os.environ["PADDLE_TRAINERS"])
        training_role = os.environ["PADDLE_TRAINING_ROLE"]

        port = os.getenv("PADDLE_PSERVER_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVER_IPS", "")
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)
        current_endpoint = os.getenv("PADDLE_CURRENT_IP", "") + ":" + port

        if training_role == "PSERVER":
            cmd += " --role pserver "
        else:
            cmd += " --role trainer "
        cmd += " --endpoints %s " % pserver_endpoints
        cmd += " --current_endpoint %s " % current_endpoint
        cmd += " --trainer_id %s " % trainer_id
        cmd += " --trainers %s " % trainers

    logging.info("run cluster commands: {}".format(cmd))

    exit(os.system(cmd))


def download():
    hadoop_home = os.getenv("HADOOP_HOME")

    configs = {}
    configs["fs.default.name"] = os.getenv("DATA_FS_NAME")
    configs["hadoop.job.ugi"] = os.getenv("DATA_FS_UGI")
    client = hdfs_utils.HDFSClient(hadoop_home, configs)

    local_train_data_dir = os.getenv("TRAIN_DATA_LOCAL", "data")
    hdfs_train_data_dir = os.getenv("TRAIN_DATA_HDFS", "")

    downloads = hdfs_utils.multi_download(client, hdfs_train_data_dir, local_train_data_dir, 0, 1, multi_processes=1)

    print(downloads)
    for d in downloads:
        base_dir = os.path.dirname(d)
        tar_cmd = "tar -zxvf {} -C {}".format(d, base_dir)
        print tar_cmd

    for d in downloads:
        base_dir = os.path.dirname(d)
        tar_cmd = "tar -zxvf {} -C {}".format(d, base_dir)
        logging.info("DOWNLOAD DATA: {}, AND TAR IT: {}".format(d, tar_cmd))
        os.system(tar_cmd)


def env_declar():
    logging.info("********  Rename Cluster Env to PaddleFluid Env ********")

    if os.environ["TRAINING_ROLE"] == "PSERVER" or os.environ["PADDLE_IS_LOCAL"] == "0":
        os.environ["PADDLE_TRAINING_ROLE"] = os.environ["TRAINING_ROLE"]
        os.environ["PADDLE_PSERVER_PORT"] = os.environ["PADDLE_PORT"]
        os.environ["PADDLE_PSERVER_IPS"] = os.environ["PADDLE_PSERVERS"]
        os.environ["PADDLE_TRAINERS"] = os.environ["PADDLE_TRAINERS_NUM"]
        os.environ["PADDLE_CURRENT_IP"] = os.environ["POD_IP"]
        os.environ["PADDLE_TRAINER_ID"] = os.environ["PADDLE_TRAINER_ID"]

    os.environ["CPU_NUM"] = os.getenv("CPU_NUM", "12")
    os.environ["NUM_THREADS"] = os.getenv("NUM_THREADS", "12")

    logging.info("Content-Type: text/plain\n\n")
    for key in os.environ.keys():
        logging.info("%30s %s \n" % (key, os.environ[key]))

    logging.info("******  Rename Cluster Env to PaddleFluid Env END ******")


if __name__ == '__main__':
    env_declar()

    if os.getenv("NEED_CUSTOM_DOWNLOAD", ""):

        if os.environ["PADDLE_TRAINING_ROLE"] == "PSERVER":
            logging.info("PSERVER do not need to download datas")
        else:
            logging.info("NEED_CUSTOM_DOWNLOAD is True, will download train data with hdfs_utils")
            download()

    run()
