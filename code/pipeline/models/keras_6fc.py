import tensorflow as tf


def create_model(input_shape=(10, )):
    inputs = tf.keras.Input(input_shape, batch_size=25)
    layer = inputs
    for i in range(6):
        layer = tf.keras.layers.Dense(20, activation='relu')(layer)
    outputs = tf.keras.layers.Dense(2, activation='sigmoid')(layer)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile('adam', 'categorical_crossentropy')
    model.summary()
    return model
