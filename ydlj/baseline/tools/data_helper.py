from os.path import join
import gzip
import pickle
import json
from tqdm import tqdm
from tools.data_iterator_pack import DataIteratorPack


class DataHelper:
    def __init__(self, gz=True, config=None):
        """
        Initialize the device.

        Args:
            self: (todo): write your description
            gz: (int): write your description
            config: (todo): write your description
        """
        self.DataIterator = DataIteratorPack
        self.gz = gz  # true
        self.suffix = '.pkl.gz' if gz else '.pkl'

        self.data_dir = config.data_dir

        self.__train_features__ = None
        self.__dev_features__ = None

        self.__train_examples__ = None
        self.__dev_examples__ = None

        self.__train_graphs__ = None
        self.__dev_graphs__ = None

        self.__train_example_dict__ = None
        self.__dev_example_dict__ = None

        self.config = config

    @property
    def sent_limit(self):   
        """
        Returns a list of sentry.

        Args:
            self: (todo): write your description
        """
        return 100  

    @property
    def entity_limit(self):
        """
        Returns the entity limit.

        Args:
            self: (todo): write your description
        """
        return 80

    @property
    def n_type(self):
        """
        Return the type : int

        Args:
            self: (todo): write your description
        """
        return 2

    
    def get_feature_file(self, tag):   
        """
        Returns a feature file with the given tag.

        Args:
            self: (str): write your description
            tag: (str): write your description
        """
        return join(self.data_dir, tag + '_feature' + self.suffix)

    def get_example_file(self, tag):
        """
        Returns the example file for the given tag.

        Args:
            self: (str): write your description
            tag: (str): write your description
        """
        return join(self.data_dir, tag + '_example' + self.suffix)


    @property
    def train_feature_file(self):
        """
        Train the feature file.

        Args:
            self: (todo): write your description
        """
        return self.get_feature_file('train')

    @property
    def dev_feature_file(self):
        """
        Return the feature file

        Args:
            self: (todo): write your description
        """
        return self.get_feature_file('dev')

    @property
    def train_example_file(self):
        """
        The training example file.

        Args:
            self: (todo): write your description
        """
        return self.get_example_file('train')

    @property
    def dev_example_file(self):
        """
        The example example example of the example.

        Args:
            self: (todo): write your description
        """
        return self.get_example_file('dev')

    @staticmethod
    def compress_pickle(pickle_file_name):
        """
        Compress a pickle file.

        Args:
            pickle_file_name: (str): write your description
        """
        def abbr(obj):
            """
            Returns a string representation of an object.

            Args:
                obj: (todo): write your description
            """
            obj_str = str(obj)
            if len(obj_str) > 100:
                return obj_str[:20] + ' ... ' + obj_str[-20:]
            else:
                return obj_str

        def get_obj_dict(pickle_obj):
            """
            Get a pickle object.

            Args:
                pickle_obj: (todo): write your description
            """
            if isinstance(pickle_obj, list):
                obj = pickle_obj[0]
            elif isinstance(pickle_obj, dict):
                obj = list(pickle_obj.values())[0]
            else:
                obj = pickle_obj
            if isinstance(obj, dict):
                return obj
            else:
                return obj.__dict__

        pickle_obj = pickle.load(open(pickle_file_name, 'rb'))

        for k, v in get_obj_dict(pickle_obj).items():
            print(k, abbr(v))
        with gzip.open(pickle_file_name + '.gz', 'wb') as fout:
            pickle.dump(pickle_obj, fout)
        pickle_obj = pickle.load(gzip.open(pickle_file_name + '.gz', 'rb'))
        for k, v in get_obj_dict(pickle_obj).items():
            print(k, abbr(v))

    def __load__(self, file):
        """
        Load pickle from file.

        Args:
            self: (todo): write your description
            file: (str): write your description
        """
        if file.endswith('json'):
            return json.load(open(file, 'r'))
        with self.get_pickle_file(file) as fin:
            print('loading', file)
            return pickle.load(fin)

    def get_pickle_file(self, file_name):
        """
        Get the pickle file.

        Args:
            self: (str): write your description
            file_name: (str): write your description
        """
        if self.gz:
            return gzip.open(file_name, 'rb')
        else:
            return open(file_name, 'rb')

    def __get_or_load__(self, name, file):
        """
        Get a pickle object by name.

        Args:
            self: (todo): write your description
            name: (str): write your description
            file: (str): write your description
        """
        if getattr(self, name) is None:   
            with self.get_pickle_file(file) as fin:
                print('loading', file)
                setattr(self, name, pickle.load(fin))

        return getattr(self, name)

    # Features
    @property
    def train_features(self):
        """
        Train the classifier.

        Args:
            self: (todo): write your description
        """
        return self.__get_or_load__('__train_features__', self.train_feature_file)

    @property
    def dev_features(self):
        """
        Return dev_features features.

        Args:
            self: (todo): write your description
        """
        return self.__get_or_load__('__dev_features__', self.dev_feature_file)

    # Examples
    @property
    def train_examples(self):
        """
        Returns the examples of the examples.

        Args:
            self: (todo): write your description
        """
        return self.__get_or_load__('__train_examples__', self.train_example_file)

    @property
    def dev_examples(self):
        """
        The sample examples.

        Args:
            self: (todo): write your description
        """
        return self.__get_or_load__('__dev_examples__', self.dev_example_file)


    # Example dict
    @property
    def train_example_dict(self):
        """
        Returns an example dictionary.

        Args:
            self: (todo): write your description
        """
        if self.__train_example_dict__ is None:
            self.__train_example_dict__ = {e.qas_id: e for e in self.train_examples}
        return self.__train_example_dict__

    @property
    def dev_example_dict(self):
        """
        A dict of dev_example classes.

        Args:
            self: (todo): write your description
        """
        if self.__dev_example_dict__ is None:
            self.__dev_example_dict__ = {e.qas_id: e for e in self.dev_examples}
        return self.__dev_example_dict__

    # Feature dict
    @property
    def train_feature_dict(self):
        """
        Return a dictionary of feature_dict features.

        Args:
            self: (todo): write your description
        """
        return {e.qas_id: e for e in self.train_features}

    @property
    def dev_feature_dict(self):
        """
        : return : class : class : dev_feature.

        Args:
            self: (todo): write your description
        """
        return {e.qas_id: e for e in self.dev_features}

    # Load
    def load_dev(self):
        """
        Load the dev_features.

        Args:
            self: (str): write your description
        """
        return self.dev_features, self.dev_example_dict#, self.dev_graphs

    def load_train(self):
        """
        Loads the examples.

        Args:
            self: (todo): write your description
        """
        return self.train_features, self.train_example_dict#, self.train_graphs



    @property
    def dev_loader(self):
        """
        Loads a model.

        Args:
            self: (todo): write your description
        """
        return self.DataIterator(*self.load_dev(),   
                                 bsz=self.config.eval_batch_size,
                                 device='cuda:{}'.format(self.config.model_gpu),
                                 sent_limit=self.sent_limit,   # 25
                                 entity_limit=self.entity_limit,
                                 sequential=True,
                                )

    @property
    def train_loader(self):
        """
        Train the model.

        Args:
            self: (todo): write your description
        """
        return self.DataIterator(*self.load_train(),   # example, feature, graph
                                 bsz=self.config.batch_size,
                                 device='cuda:{}'.format(self.config.model_gpu),   
                                 sent_limit=self.sent_limit,
                                 entity_limit=self.entity_limit,
                                 sequential=False
            )


