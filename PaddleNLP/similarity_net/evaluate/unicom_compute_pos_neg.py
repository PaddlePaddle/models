"""
comput unicom
"""

infer_results = []
labels = []
result = []
temp_reuslt = []
temp_query = ""
pos_num = 0.0
neg_num = 0.0

with open("./unicom_infer_result", "r") as infer_result_file:
    for line in infer_result_file:
        infer_results.append(line.strip().split("\t"))

with open("./unicom_label", "r") as label_file:
    for line in label_file:
        labels.append(line.strip().split("\t"))

for infer_result, label in zip(infer_results, labels):
    if infer_result[0] != temp_query and temp_query != "":
        result.append(temp_reuslt)
        temp_query = infer_result[0]
        temp_reuslt = []
        temp_reuslt.append(infer_result + label)
    else:
        if temp_query == '':
            temp_query = infer_result[0]
        temp_reuslt.append(infer_result + label)
else:
    result.append(temp_reuslt)

for _result in result:
    for n, i in enumerate(_result, start=1):
        for j in _result[n:]:
            if (int(j[-1]) > int(i[-1]) and float(j[-2]) < float(i[-2])) or (
                    int(j[-1]) < int(i[-1]) and float(j[-2]) > float(i[-2])):
                neg_num += 1
            elif (int(j[-1]) > int(i[-1]) and float(j[-2]) > float(i[-2])) or (
                    int(j[-1]) < int(i[-1]) and float(j[-2]) < float(i[-2])):
                pos_num += 1

print("pos/neg of unicom data is %f" % (pos_num / neg_num))
