import tensorflow as tf
from tensorflow.python import keras
from randnet.model.randnet import RandNetSmall


def train():
    regularizer = keras.regularizers.l2(0.0001)
    model = RandNetSmall(10, kernel_regularizer=regularizer, bias_regularizer=regularizer)
    optimizer = keras.optimizers.Adam(0.0004)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="log_dir", write_images=True)

    model.fit(x_train,
              y_train,
              batch_size=32,
              epochs=100,
              callbacks=[tensorboard_callback],
              validation_data=(x_test, y_test))
    model.save_weights("model_weights")


if __name__ == '__main__':
    train()
