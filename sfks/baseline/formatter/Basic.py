class BasicFormatter:
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        return data

