import json
import os
import random

input_path = "/input"
output_path = "/output/result.txt"

if __name__ == "__main__":
    result = {}
    for filename in os.listdir(input_path):
        data = []
        for line in open(os.path.join(input_path, filename), "r", encoding="utf8"):
            data.append(json.loads(line))
        for item in data:
            id = item["id"]
            result[id] = []
            for option in ["A", "B", "C", "D"]:
                if random.randint(1, 2) == 1:
                    result[id].append(option)

    json.dump(result, open(output_path, "w", encoding="utf8"), indent=2, ensure_ascii=False, sort_keys=True)
