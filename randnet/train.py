import os
import math
import argparse


from tensorflow.python import keras

from randnet.data.loader import DataLoader
from randnet.data.mapper import DataSetMapper
from randnet.model.randnet import RandNetSmall


def half_cosine_lr_schedule(epoch, total_n_epochs=100, initial_lr=0.1):
    x = (epoch / float(total_n_epochs)) * math.pi
    return initial_lr * 0.5 * (math.cos(x) + 1)


class TensorboardCallbackWithLR(keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def train(experiment_dir="experiment",
          dataset="cifar10",
          batch_size=32,
          epochs=100,
          l2=0.0001,
          initial_lr=0.0004):
    data_set_mapper = DataSetMapper(dataset)

    data_loader = DataLoader(data_set_mapper.name,
                             batch_size,
                             train_split=data_set_mapper.train_split,
                             val_split=data_set_mapper.val_split)

    regularizer = keras.regularizers.l2(l2)
    model = RandNetSmall(data_loader.num_classes, kernel_regularizer=regularizer, bias_regularizer=regularizer)
    optimizer = keras.optimizers.Adam(initial_lr)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[keras.metrics.accuracy,
                 keras.metrics.top_k_categorical_accuracy])

    # model.build(data_loader.shape)
    # model.summary()

    log_dir = os.path.join(experiment_dir, "logs")
    tensorboard_callback = TensorboardCallbackWithLR(log_dir=log_dir, write_images=True)

    train_iterator = data_loader.train_one_shot_iterator
    val_iterator = data_loader.val_one_shot_iterator

    learning_rate_scheduler = keras.callbacks.LearningRateScheduler(
        lambda x: half_cosine_lr_schedule(x, initial_lr=initial_lr, total_n_epochs=epochs),
        verbose=1)

    model.fit(train_iterator,
              steps_per_epoch=data_loader.train_steps_per_epoch,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=val_iterator,
              callbacks=[tensorboard_callback, learning_rate_scheduler],
              validation_steps=data_loader.val_steps_per_epoch)

    weight_path = os.path.join(experiment_dir, "weights/model_weights")
    model.save_weights(weight_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Keras Randomly Wired Networks Training")
    parser.add_argument("--experiment-dir", default="experiment", type=str)
    parser.add_argument("--dataset", default="cifar10", choices=list(DataSetMapper.VAL_SPLIT_MAPPING.keys()), type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--l2", default=0.0001, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--initial-lr", default=0.1, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(
        experiment_dir=args.experiment_dir,
        dataset=args.dataset,
        epochs=args.epochs,
        l2=args.l2,
        batch_size=args.batch_size,
        initial_lr=args.initial_lr
    )
