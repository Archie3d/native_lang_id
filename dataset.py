import tensorflow as tf
import tensorflow_text as tftext
import os
import numpy as np

class TextSequence(tf.keras.utils.Sequence):

    def __init__(self, path, countries, batch_size, tokenizer, shuffle=False, limit=None, epsilon=0.01):
        self.path = path
        self.countries = countries
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.limit = limit
        self.epsilon = epsilon
        self.min_words = 5
        self.max_words = 100

        self.classified_text = []
        self.indices = []

        self.build_index()

        if self.shuffle:
            self.shuffle_indices()


    def __len__(self):
        n = len(self.indices) // self.batch_size

        if self.limit is not None:
            return min(self.limit, n)

        return n


    def __getitem__(self, index):
        lines = []
        labels = []

        for i in range(self.batch_size):
            idx = self.indices[index * self.batch_size + i]
            line, label = self.classified_text[idx]
            lines.append(line)
            labels.append(label)

        tokens = self.tokenizer.tokenize(lines).merge_dims(-2, -1).to_tensor()
        labels = np.concatenate(labels, axis=0)

        return tokens, labels


    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_indices()


    def shuffle_indices(self):
        np.random.shuffle(self.indices)


    def build_index(self):
        index = 0

        for file in os.listdir(self.path):
            country = os.path.splitext(file)[0]

            print(f"Scanning {country}...      ", end='\r')

            if not self.is_country_in_dataset(country):
                print(f"Skipped {country}              ")
                continue

            in_file_path = os.path.join(self.path, file)

            with open(in_file_path, mode="r", encoding="utf-8") as f:
                while True:
                    line = f.readline().strip()

                    if not line:
                        break

                    # Ignore too short or too long text
                    n_words = len(line.split())

                    if n_words < self.min_words or n_words > self.max_words:
                        continue

                    self.classified_text.append((line, self.make_label_tensor(country)))
                    self.indices.append(index)
                    index +=1

        print("\n")
        print(f"Dataset contains {len(self.indices)} sentences.")


    def is_country_in_dataset(self, country):
        for c in self.countries:
            if c == country:
                return True

        return False


    def make_label_tensor(self, country):
        N = len(self.countries)
        tensor = np.full(shape=(1, N), fill_value=self.epsilon)

        one = 1.0 - self.epsilon * (N - 1)

        for i in range(len(self.countries)):
            if country == self.countries[i]:
                tensor[0, i] = one
                break

        return tensor
