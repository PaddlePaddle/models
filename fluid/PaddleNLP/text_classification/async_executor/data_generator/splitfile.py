"""
Split file into parts
"""
import sys
import os
block = int(sys.argv[1])
datadir = sys.argv[2]
file_list = []
for i in range(block):
    file_list.append(open(datadir + "/part-" + str(i), "w"))
id_ = 0
for line in sys.stdin:
    file_list[id_ % block].write(line)
    id_ += 1
for f in file_list:
    f.close()
