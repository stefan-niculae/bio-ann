import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Dropout

from connections_generation import current_cond_dists, generate_masks
from bio_do import CustomConnections

# fashion mnist properties
N_PIXELS = 28*28
N_CLASSES = 10


def keras_example() -> Model:
    """
    https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
    98.4% test acc in 20 epochs (40s)
    """
    model = Sequential([
        Dense(512, activation='relu'),
        Dropout(.2),
        Dense(512, activation='relu'),
        Dropout(.2),
        Dense(N_CLASSES, activation='softmax'),
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def no_dropout(layer_sizes=(128,)*3) -> Model:
    """
    """
    model = Sequential()
    for layer_size in layer_sizes:
        model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(N_CLASSES, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def classic_dropout(layer_sizes=(128,)*3, do_proba=.4) -> Model:
    """
    ignoring the entire neuron, a random one each time
    """
    model = Sequential()
    for layer_size in layer_sizes:
        model.add(Dense(layer_size, activation='relu'))
        model.add(Dropout(do_proba))
    model.add(Dense(N_CLASSES, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def static_dropout(layer_sizes=(128,)*3, do_proba=.4) -> Model:
    """
    ignoring the entire neuron, the same time every time
    """
    masks = [
        np.random.random(layer_size)[np.newaxis] > do_proba
        for layer_size in layer_sizes
    ]

    model = Sequential()
    for i, layer_size in enumerate(layer_sizes):
        model.add(Dense(layer_size, activation='relu'))
        model.add(Lambda(lambda x: x * masks[i]))
    model.add(Dense(N_CLASSES, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def our_dropout(layer_sizes=(128,)*3, masks=None) -> Model:
    """
    ignoring some neuron connections, and over time adding more
    """
    if masks is None:
        masks = generate_masks([N_PIXELS, *layer_sizes, N_CLASSES], current_cond_dists())
    # TODO turn masks into a tensor to make it fast

    model = Sequential()
    for i, layer_size in enumerate(layer_sizes):
        model.add(CustomConnections(layer_size, masks[i], activation='relu',
                                    input_dim=N_PIXELS if i == 0 else None))

    model.add(CustomConnections(N_CLASSES, masks[-1], activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
