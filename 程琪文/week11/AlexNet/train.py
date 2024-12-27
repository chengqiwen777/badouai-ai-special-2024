import numpy as np
import cv2
from model.AlexNet import AlexNet
from keras import backend as K
import utils
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

K.set_image_data_format('channels_first')

def generate_arrays_from_file(lines, batch_size):
    n = len(lines)
    i = 0
    while True:
        X_train = []
        Y_train = []

        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            img = cv2.imread(name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            i = (i + 1) % n

        X_train = utils.resize_images(X_train)
        X_train = X_train.reshape(-1, 224, 224, 3)
        Y_train = to_categorical(Y_train, num_classes=2)

        yield (X_train, Y_train)


if __name__ == '__main__':
    logs_dir = 'logs'

    with open('dataset.txt', 'r') as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    model = AlexNet()

    model_checkpoints = ModelCheckpoint(logs_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                        monitor='acc',
                                        save_weights_only=False,
                                        save_best_only=True,
                                        period=3)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                                  factor=0.5,
                                  patience=5,
                                  verbose=1)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   verbose=1,
                                   min_delta=0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])

    batch_size = 128

    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size=batch_size),
                        steps_per_epoch=max(1, num_train//batch_size),
                        epochs=50,
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size=batch_size),
                        validation_steps=max(1, num_val//batch_size),
                        callbacks=[model_checkpoints, reduce_lr],
                        initial_epoch=0)
    model.save_weights(logs_dir + 'model.h5')
