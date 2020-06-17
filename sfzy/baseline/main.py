# coding=utf-8

import json
import re

input_path = "/input/input.json"
output_path = "/output/result.json"


def get_summary(text):
    for i, _ in enumerate(text):
        sent_text = text[i]["sentence"]
        if re.search(r"诉讼请求：", sent_text):
            text0 = text[i]["sentence"]
            text1 = text[i + 1]["sentence"]
            text2 = text[i + 2]["sentence"]
            break
        else:
            text0 = text[11]["sentence"]
            text1 = text[12]["sentence"]
            text2 = text[13]["sentence"]
    result = text0 + text1 + text2
    return result


if __name__ == "__main__":
    with open(output_path, 'a', encoding='utf8') as fw:
        with open(input_path, 'r', encoding="utf8") as f:
            for line in f:
                data = json.loads(line)
                id = data.get('id')
                text = data.get('text')  # "text": [{"sentence":"001"},{"sentence":"002"}]
                summary = get_summary(text)  # your model predict
                result = dict(
                    id=id,
                    summary=summary
                )
                fw.write(json.dumps(result, ensure_ascii=False) + '\n')
