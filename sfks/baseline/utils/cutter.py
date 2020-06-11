import json
import jieba
import os
import argparse

frequency = {}
word2id = {"PAD": 0, "UNK": 1}
min_freq = 10


def cut(s):
    arr = list(jieba.cut(s))
    for word in arr:
        if word not in frequency:
            frequency[word] = 0
        frequency[word] += 1

    return arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--gen_word2id', action="store_true")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        fin = open(os.path.join(input_path, filename), "r", encoding="utf8")
        fout = open(os.path.join(output_path, filename), "w", encoding="utf8")

        for line in fin:
            data = json.loads(line)
            data["statement"] = cut(data["statement"])
            for option in ["A", "B", "C", "D"]:
                data["option_list"][option] = cut(data["option_list"][option])

            print(json.dumps(data, ensure_ascii=False, sort_keys=True), file=fout)

    if args.gen_word2id:
        for word in frequency:
            if frequency[word] >= min_freq:
                word2id[word] = len(word2id)
        json.dump(word2id, open("../data/word2id.txt", "w", encoding="utf8"), indent=2, ensure_ascii=False)
