import json
import os
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search


class JsonFromFilesDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding

        filename_list = config.get("data", "%s_file_list" % mode).replace(" ", "").split(",")

        self.data = []
        for filename in filename_list:
            f = open(os.path.join(self.data_path, filename), "r", encoding="utf8")
            for line in f:
                self.data.append(json.loads(line))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
