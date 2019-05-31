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

    model.build(data_loader.shape)
    model.summary()

    train_iterator = data_loader.train_one_shot_iterator
    val_iterator = data_loader.val_one_shot_iterator
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="log_dir", write_images=True)

    model.fit(train_iterator,
              steps_per_epoch=data_loader.train_steps_per_epoch,
              epochs=10,
              batch_size=batch_size,
              validation_data=val_iterator,
              callbacks=[tensorboard_callback],
              validation_steps=data_loader.val_steps_per_epoch)

    model.save_weights("model_weights/weights")


if __name__ == '__main__':
    train()
