from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def AlexNet(input_shape=(224, 224, 3), num_classes=2):
    model = Sequential()

    model.add(
        Conv2D(filters=48,
               kernel_size=(11, 11),
               strides=(4, 4),
               padding='valid',
               activation='relu',
               input_shape=input_shape)
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2)))

    model.add(Conv2D(filters=128,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     padding='same',
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2)))

    model.add(Conv2D(filters=192,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=192,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='same',
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    return model
