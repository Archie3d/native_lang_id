import tensorflow as tf
import tensorflow_text as tftext

countries = [
    'Albania',
    'Argentina',
    'Armenia',
    'Australia',
    'Austria',
    'Belarus',
    'Belgium',
    'Bosnia',
    'Brazil',
    'Bulgaria',
    'Canada',
    'Chile',
    'China',
    'Colombia',
    'Croatia',
    'Cyprus',
    'Czech',
    'Denmark',
    'Estonia',
    'Finland',
    'France',
    'Germany',
    'Greece',
    'Hungary',
    'Iceland',
    'India',
    'Ireland',
    'Israel',
    'Italy',
    'Latvia',
    'Lithuania',
    'Netherlands',
    'Norway',
    'Poland',
    'Portugal',
    'Romania',
    'Russia',
    'Serbia',
    'Slovakia',
    'Slovenia',
    'Spain',
    'Sweden',
    'Switzerland',
    'Turkey',
    'UK',
    'Ukraine',
    'US'
]

def get_country_index(country):
    """
    Returns index of the country in the countries[] list
    """
    for i in range(len(countries)):
        if countries[i] == country:
            return i

    return None


def load_vocabulary(path):
    """
    Load vocabulary from a text file.
    """
    vocab = []

    with open(path, mode="r", encoding="utf-8") as f:
        while True:
            line = f.readline().strip()

            if not line:
                break

            vocab.append(line)

    return vocab


def make_bert_tokenizer(vocab, lower_case=False):
    """
    Construct a BERT tokenizer for a given vocabulary.
    """
    vocab_values = tf.range(tf.size(vocab, out_type=tf.int64), dtype=tf.int64)
    init = tf.lookup.KeyValueTensorInitializer(keys=vocab, values=vocab_values, key_dtype=tf.string, value_dtype=tf.int64)
    num_oov = 1
    vocab_table = tf.lookup.StaticVocabularyTable(init, num_oov, lookup_key_dtype=tf.string)

    return tftext.BertTokenizer(vocab_table, token_out_type=tf.int32, lower_case=lower_case)
