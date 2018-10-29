input = open("list.txt", "r").readlines()
fout_train = open("CUB200_train.txt", "w")
fout_valid = open("CUB200_val.txt", "w")
for i, item in enumerate(input):
    label = item.strip().split("/")[-2].split(".")[0]
    label = int(label)
    if label <= 100:
        fout = fout_train
    else:
        fout = fout_valid
    fout.write(item.strip() + " " + str(label) + "\n")

fout_train.close()
fout_valid.close()
