import sys


def convert_format(input, output):
    with open(input) as rf:
        with open(output, "w") as wf:
            last_sess = -1
            sign = 1
            i = 0
            for l in rf:
                i = i + 1
                if i == 1:
                    continue
                if (i % 1000000 == 1):
                    print(i)
                tokens = l.strip().split()
                if (int(tokens[0]) != last_sess):
                    if (sign):
                        sign = 0
                        wf.write(tokens[1] + " ")
                    else:
                        wf.write("\n" + tokens[1] + " ")
                    last_sess = int(tokens[0])
                else:
                    wf.write(tokens[1] + " ")


input = "rsc15_train_tr.txt"
output = "rsc15_train_tr_paddle.txt"
input2 = "rsc15_test.txt"
output2 = "rsc15_test_paddle.txt"
convert_format(input, output)
convert_format(input2, output2)
