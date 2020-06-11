import json
import os
import jieba

data_path = "data/"
test_path = "/input"
output_path = "/output"

if __name__ == "__main__":
    os.system(
        "sudo /home/user/miniconda/bin/python3 utils/cutter.py --input /input --output data/cutted")
    os.system(
        "sudo /home/user/miniconda/bin/python3 test.py --gpu 0 --config config/model.config --checkpoint data/model/model/31.pkl --result result.json")
    result = {}
    for item in json.load(open("result.json", "r", encoding="utf8")):
        result[item["id"]] = item["answer"]
    json.dump(result, open("/output/result.txt", "w", encoding="utf8"), indent=2, ensure_ascii=False, sort_keys=True)
