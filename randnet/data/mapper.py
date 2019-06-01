import tensorflow_datasets as tfds


class DataSetMapper:
    VAL_SPLIT_MAPPING = {
        "imagenet2012": tfds.Split.VALIDATION,
        "cifar10": tfds.Split.TEST,
        "cifar100": tfds.Split.TEST,
        "fashion_mnist": tfds.Split.TEST,
        "mnist": tfds.Split.TEST
    }

    def __init__(self, name):
        if name not in DataSetMapper.VAL_SPLIT_MAPPING.keys():
            raise ValueError("There is no mapping for data set name {}. Available ones: {}".format(
                name,
                DataSetMapper.VAL_SPLIT_MAPPING.keys()
            ))
        self._name = name

    @property
    def train_split(self):
        return tfds.Split.TRAIN

    @property
    def val_split(self):
        return DataSetMapper.VAL_SPLIT_MAPPING[self._name]

    @property
    def name(self):
        return self._name
