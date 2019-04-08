import os

vallist = 'vallist.txt'
testlist = 'testlist.txt'
sampling_times = 10
cropping_times = 3

fl = open(vallist).readlines()
fl = [line.strip() for line in fl if line.strip() != '']
f_test = open(testlist, 'w')

for i in range(len(fl)):
    line = fl[i].split(' ')
    fn = line[0]
    label = line[1]
    for j in range(sampling_times):
        for k in range(cropping_times):
            test_item = fn + ' ' + str(i) + ' ' + str(j) + ' ' + str(k) + '\n'
            f_test.write(test_item)

f_test.close()
