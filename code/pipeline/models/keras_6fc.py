import tensorflow as tf


def create_model(input_shape):

    inputs = tf.keras.Input(input_shape)
    layer = inputs

    # 6 layers
    for i in range(6):
        layer = tf.keras.layers.Dense(32, activation='relu')(layer)

    # 13 outputs expected
    outputs = tf.keras.layers.Dense(13, activation='softmax')(layer)

    # build model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # optimizer:adam ; lossfunc:categorical_crossentropy
    model.compile('adam', 'categorical_crossentropy')
    model.summary()

    return model
