#!/usr/bin/env python
# -*- coding: utf-8 -*- 
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: conversation_client.py
"""

from __future__ import print_function

import sys
import socket

reload(sys)
sys.setdefaultencoding('utf8')

SERVER_IP = "127.0.0.1"
SERVER_PORT = 8601

def conversation_client(text):
    """
    conversation_client
    """
    mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    mysocket.connect((SERVER_IP, SERVER_PORT))

    mysocket.sendall(text.encode())
    result = mysocket.recv(4096).decode()

    mysocket.close()

    return result


def main():
    """
    main
    """
    if len(sys.argv) < 2:
        print("Usage: " + sys.argv[0] + " eval_file")
        exit()

    for line in open(sys.argv[1]):
        response = conversation_client(line.strip())
        print(response)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
