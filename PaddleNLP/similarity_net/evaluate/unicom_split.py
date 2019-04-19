"""
split unicom file
"""

with open("../data/unicom", "r") as unicom_file:
    with open("./unicom_infer", "w") as infer_file:
        with open("./unicom_label", "w") as label_file:
            for line in unicom_file:
                line = line.strip().split('\t')
                infer_file.write("\t".join(line[:2]) + '\n')
                label_file.write(line[2] + '\n')
