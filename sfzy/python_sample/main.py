import json

input_path = '/input/input.json'
output_path = '/output/result.json'

if __name__ == "__main__":
    with open(output_path, 'w', encoding='utf8') as fw:
        with open(input_path, 'r', encoding="utf8") as f:
            for line in f:
                data = json.loads(line)
                id = data.get('id')
                text = data.get('text')  # "text": [{"sentence":"001"},{"sentence":"002"}]
                summary = '司法摘要的内容'  # your model predict
                result = dict(
                    id=id,
                    summary=summary
                )
                fw.write(json.dumps(result, ensure_ascii=False) + '\n')
