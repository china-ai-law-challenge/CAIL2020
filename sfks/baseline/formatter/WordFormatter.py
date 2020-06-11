import json
import torch
import numpy as np
import os
from pytorch_pretrained_bert import BertTokenizer


class WordFormatter:
    def __init__(self, config, mode):
        self.max_question_len = config.getint("data", "max_question_len")
        self.max_option_len = config.getint("data", "max_option_len")

        self.word2id = json.load(open(config.get("data", "word2id"), "r", encoding="utf8"))

    def convert_tokens_to_ids(self, tokens):
        arr = []
        for a in range(0, len(tokens)):
            if tokens[a] in self.word2id:
                arr.append(self.word2id[tokens[a]])
            else:
                arr.append(self.word2id["UNK"])
        return arr

    def convert(self, tokens, l, bk=False):
        while len(tokens) < l:
            tokens.append("PAD")
        if bk:
            tokens = tokens[len(tokens) - l:]
        else:
            tokens = tokens[:l]
        ids = self.convert_tokens_to_ids(tokens)

        return ids

    def process(self, data, config, mode, *args, **params):
        context = []
        question = []
        label = []
        idx = []

        for temp_data in data:
            idx.append(temp_data["id"])

            if mode != "test":
                label_x = 0
                if "A" in temp_data["answer"]:
                    label_x += 1
                if "B" in temp_data["answer"]:
                    label_x += 2
                if "C" in temp_data["answer"]:
                    label_x += 4
                if "D" in temp_data["answer"]:
                    label_x += 8
                label.append(label_x)

            temp_context = []
            temp_question = []

            for option in ["A", "B", "C", "D"]:
                temp_question.append(self.convert(temp_data["statement"], self.max_question_len, bk=True))
                temp_context.append(self.convert(temp_data["option_list"][option], self.max_option_len))

            context.append(temp_context)
            question.append(temp_question)

        question = torch.LongTensor(question)
        context = torch.LongTensor(context)
        if mode != "test":
            label = torch.LongTensor(np.array(label, dtype=np.int32))
            return {"context": context, "question": question, 'label': label, "id": idx}
        else:
            return {"context": context, "question": question, "id": idx}
