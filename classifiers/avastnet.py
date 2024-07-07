import logging

from keras import Input, Model, metrics
from keras.layers import Layer, Embedding, Convolution1D, GlobalAveragePooling1D, Dense, MaxPooling1D


AVASTNET_MAX_INPUT_LENGTH = 512000
VOCABULARY_SIZE = 257
EPOCHS = 10
BATCH_SIZE = 32


def make_avastnet():
    input_layer = Input((AVASTNET_MAX_INPUT_LENGTH,), batch_size=BATCH_SIZE, dtype="int32")

    embedding = Embedding(input_dim=VOCABULARY_SIZE, output_dim=8)(input_layer)

    conv_32_1 = Convolution1D(filters=48, kernel_size=32, strides=4,
                              padding="same", activation="relu")(embedding)
    conv_32_2 = Convolution1D(filters=96, kernel_size=32, strides=4,
                              padding="same", activation="relu")(conv_32_1)
    max_pooling = MaxPooling1D(pool_size=4)(conv_32_2)

    conv_16_1 = Convolution1D(filters=128, kernel_size=16, strides=8,
                              padding="same", activation="relu")(max_pooling)
    conv_16_2 = Convolution1D(filters=24, kernel_size=16, strides=8,
                              padding="same", activation="relu")(conv_16_1)

    avg_pooling = GlobalAveragePooling1D()(conv_16_2)

    dense_1 = Dense(192, activation="selu")(avg_pooling)
    dense_2 = Dense(160, activation="selu")(dense_1)
    dense_3 = Dense(128, activation="selu")(dense_2)

    output = Dense(1, activation="sigmoid")(dense_3)

    avast_net = Model(inputs=input_layer, outputs=output)

    compile_avast(avast_net)

    logging.info(avast_net.summary())

    return avast_net


def compile_avast(avast_net):
    avast_metrics = [metrics.binary_accuracy]
    avast_net.compile(loss='binary_crossentropy', optimizer="adam", metrics=avast_metrics)
