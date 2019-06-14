#!/usr/bin/env python
# -*- coding: utf-8 -*- 
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: conversation_server.py
"""

from __future__ import print_function

import sys
sys.path.append("../")
import socket
from thread import start_new_thread
from tools.conversation_strategy import load
from tools.conversation_strategy import predict

reload(sys)
sys.setdefaultencoding('utf8')

SERVER_IP = "127.0.0.1"
SERVER_PORT = 8601

print("starting conversation server ...")
print("binding socket ...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#Bind socket to local host and port
try:
    s.bind((SERVER_IP, SERVER_PORT))
except socket.error as msg:
    print("Bind failed. Error Code : " + str(msg[0]) + " Message " + msg[1])
    exit()
#Start listening on socket
s.listen(10)
print("bind socket success !")

print("loading model...")
model = load()
print("load model success !")

print("start conversation server success !")


def clientthread(conn, addr):
    """
    client thread
    """
    logstr = "addr:" + addr[0]+ "_" + str(addr[1])
    try:
        #Receiving from client
        param = conn.recv(4096).decode()
        logstr += "\tparam:" + param 
        if param is not None:
            response = predict(model, param.strip())
            logstr += "\tresponse:" + response 
            conn.sendall(response.encode())
        conn.close()
        print(logstr + "\n")
    except Exception as e:
        print(logstr + "\n", e)


while True:
    conn, addr = s.accept()
    start_new_thread(clientthread, (conn, addr))
s.close()
