import json
import os
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search


class JsonFromFilesDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        """
        Initialize the config file.

        Args:
            self: (todo): write your description
            config: (dict): write your description
            mode: (todo): write your description
            encoding: (str): write your description
            params: (dict): write your description
        """
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
        """
        Get the value of item.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        return self.data[item]

    def __len__(self):
        """
        Returns the length of the data.

        Args:
            self: (todo): write your description
        """
        return len(self.data)
