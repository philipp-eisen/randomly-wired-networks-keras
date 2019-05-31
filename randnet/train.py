from tensorflow.python import keras

from randnet.dataloader import DataLoader
from randnet.model.randnet import RandNetSmall


def train(dataset="cifar10", batch_size=32):

    data_loader = DataLoader(dataset, batch_size)

    regularizer = keras.regularizers.l2(0.0001)
    model = RandNetSmall(data_loader.num_classes, kernel_regularizer=regularizer, bias_regularizer=regularizer)
    optimizer = keras.optimizers.Adam(0.0004)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    train_iterator = data_loader.train_one_shot_iterator
    val_iterator = data_loader.val_one_shot_iterator

    model.fit(train_iterator,
              steps_per_epoch=data_loader.train_steps_per_epoch,
              epochs=10,
              batch_size=batch_size,
              validation_data=val_iterator,
              validation_steps=data_loader.val_steps_per_epoch)


if __name__ == '__main__':
    train()
