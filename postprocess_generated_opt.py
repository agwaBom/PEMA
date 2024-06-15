import re
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./pema_90_0/lm_pred.txt")
    parser.add_argument("--output", type=str, default="./pema_90_0/lm_pred_post.txt")
    args = parser.parse_args()

    # remove I'm not sure and following
    data = open(args.input, "r").readlines()
    for i in range(len(data)):
        data[i] = re.sub(r' I\'m not sure.*', "", data[i])
        data[i] = re.sub(r' I\'m" .*', "", data[i])
        data[i] = re.sub(r' I\'m.*', "", data[i])
        data[i] = re.sub(r' I 50%.*', "", data[i])
        data[i] = re.sub(r'Convert the following informal sentence .*Formal:', "", data[i])
        data[i] = re.sub(r'Translate this from English to German:.*', "", data[i])
        data[i] = re.sub(r' I\..*', "", data[i])
        data[i] = re.sub(r' \.\.\.\.\.\..*', "", data[i])

    for _ in range(1000):
        for i in range(len(data)):
            data[i] = re.sub(r'\b(\w+)[ ]\1\b', r'\1', data[i])

    if data[-1] == '':
        data.pop(-1)

    if data[-1][-1] == '\n':
        data[-1] = data[-1][:-1]

    with open(args.output, "w") as f:
        f.writelines(data)
