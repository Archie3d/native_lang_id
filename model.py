import tensorflow as tf
import common

# @see https://www.tensorflow.org/text/tutorials/text_classification_rnn
# @see https://www.tensorflow.org/text/tutorials/classify_text_with_bert


def make_model(vocabulary_size, embedding_size=16, lstm_size=16, hidden_size=64):
    identification_size = len(common.countries)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, ), dtype=tf.int32),
        tf.keras.layers.Embedding(vocabulary_size + 1, embedding_size),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_size, return_sequences=False)
        ),
        tf.keras.layers.Dense(
            units=hidden_size,
            activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(
            units=identification_size,
            activation='softmax'
        )
    ])

    return model
