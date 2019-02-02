import sys

# kinetics-400_train.csv should be down loaded first and set as sys.argv[1]
# sys.argv[2] can be set as kinetics400_label.txt
# python generate_label.py kinetics-400_train.csv kinetics400_label.txt

num_classes = 400

fname = sys.argv[1]
outname = sys.argv[2]
fl = open(fname).readlines()
fl = fl[1:]
outf = open(outname, 'w')

label_list = []
for line in fl:
    label = line.strip().split(',')[0].strip('"')
    if label in label_list:
        continue
    else:
        label_list.append(label)

assert len(label_list
           ) == num_classes, "there should be {} labels in list, but ".format(
               num_classes, len(label_list))

label_list.sort()
for i in range(num_classes):
    outf.write('{} {}'.format(label_list[i], i) + '\n')

outf.close()
