import sys

precision_list = []
recall_list = []
f1_list = []
train_precision_list = []
train_recall_list = []
train_f1_list = []
for line in sys.stdin:
    line = line.strip()
    if line.startswith("[TestSet]"):
        tokens = line.split(" ")
        for token in tokens:
            field_value = token.split(":")
            field = field_value[0].strip()
            if len(field_value) != 2:
                continue
            value = float(field_value[1].strip("[] "))
            if (field == "pass_precision"):
                precision_list.append(value)
            if field == "pass_recall":
                recall_list.append(value)
            if field == "pass_f1_score":
                f1_list.append(value)
            
    elif line.startswith("[TrainSet]"):
        tokens = line.split(" ")
        for token in tokens:
            field_value = token.split(":")
            if len(field_value) != 2:
                continue
            field = field_value[0].strip()
            value = float(field_value[1].strip("[] "))
            if (field == "pass_precision"):
                train_precision_list.append(value)
            if field == "pass_recall":
                train_recall_list.append(value)
            if field == "pass_f1_score":
                train_f1_list.append(value)
assert len(precision_list) == len(recall_list)
assert len(recall_list) == len(f1_list)
assert len(train_precision_list) == len(train_recall_list)
assert len(train_recall_list) == len(train_f1_list)

for i in xrange(len(precision_list)):
    print str(precision_list[i]) + "\t" + str(recall_list[i]) + "\t" + str(f1_list[i]) + "\t" + str(train_precision_list[i]) + "\t" + str(train_recall_list[
i]) + "\t" + str(train_f1_list[i])
