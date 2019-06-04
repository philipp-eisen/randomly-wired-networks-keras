import tensorflow as tf
import tensorflow_datasets as tfds


class DataLoader:
    def __init__(self, name, batch_size,
                 train_split=tfds.Split.TRAIN,
                 val_split=tfds.Split.VALIDATION,
                 prefetch=tf.data.experimental.AUTOTUNE,
                 shuffle_buffer_size=2048,
                 label_smooth_factor=0.1):

        self.name = name
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.shuffle_buffer_size = shuffle_buffer_size

        self.val_split = val_split
        self.train_split = train_split

        self.dataset_builder = tfds.builder(name)
        self.dataset_builder.download_and_prepare()

        self.label_smoothing_factor = label_smooth_factor

    def _smooth_labels(self, label):
        assert len(label.shape) == 1
        if 0 <= self.label_smoothing_factor <= 1:
            # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
            label *= 1 - self.label_smoothing_factor
            label += self.label_smoothing_factor / label.shape[0].value
        else:
            raise Exception('Invalid label smoothing factor: ' + str(self.label_smoothing_factor))
        return label

    def _preprocess(self, image, label):
        image /= 255
        label = tf.one_hot(label, self.num_classes)
        if self.label_smoothing_factor:
            label = self._smooth_labels(label)
        return image, label

    @property
    def shape(self):
        return tf.TensorShape((self.batch_size, *self.info.features["image"].shape))

    @property
    def info(self):
        return self.dataset_builder.info

    @property
    def num_classes(self):
        return self.info.features["label"].num_classes

    @property
    def num_train_examples(self):
        return self.info.splits[tfds.Split.TRAIN].num_examples

    @property
    def num_val_examples(self):
        return self.info.splits[tfds.Split.TEST].num_examples

    @property
    def train_steps_per_epoch(self):
        return self.num_train_examples // self.batch_size

    @property
    def val_steps_per_epoch(self):
        return self.num_val_examples // self.batch_size

    @property
    def train_dataset(self):
        dataset = self.dataset_builder.as_dataset(split=self.train_split, as_supervised=True)
        dataset = dataset.map(self._preprocess)
        dataset = dataset.shuffle(self.shuffle_buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch)
        dataset = dataset.repeat()
        return dataset

    @property
    def val_dataset(self):
        dataset = self.dataset_builder.as_dataset(split=self.val_split, as_supervised=True)
        dataset = dataset.map(self._preprocess)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch)
        dataset = dataset.repeat()
        return dataset

    @property
    def train_one_shot_iterator(self):
        return self.train_dataset.make_one_shot_iterator()

    @property
    def val_one_shot_iterator(self):
        return self.val_dataset.make_one_shot_iterator()
