import sys
import common
import config
import model
import yaml
import numpy as np
from reddit import RedditClient


if len(sys.argv) < 2:
    print(f"USAGE: {sys.argv[0]} <reddit user id>")
    exit()

target_user = sys.argv[1];

with open('inference.yaml', 'r') as f:
    inference_config = yaml.safe_load(f)

reddit = RedditClient()

lower_case = True

vocab = common.load_vocabulary("./dataset/vocab.txt")
tokenizer = common.make_bert_tokenizer(vocab, lower_case=lower_case)

print(f"ℹ️ Logging to Reddit as ${inference_config['user']}...")
reddit.login(
    username=inference_config['user'],
    password=inference_config['password'],
    client_id=inference_config['client_id'],
    client_secret=inference_config['client_secret']
)

if not reddit.is_logged_in:
    raise RuntimeError("☠️ Unable to logging to Reddit")

print("💬 Fetching comments...")
text = reddit.redditor_new_comments(target_user, limit=None)

print(f"ℹ️ Got {len(text)} comments.")

reddit.logout()

model = model.make_model(
    vocabulary_size=len(vocab),
    embedding_size=config.model['embedding_size'],
    lstm_size=config.model['lstm_size'],
    hidden_size=config.model['hidden_size']
)

model.load_weights(inference_config['checkpoint'])

tokens = tokenizer.tokenize(text).merge_dims(-2, -1).to_tensor()

batch_size = 32
num_batches = tokens.shape[0] // batch_size

if num_batches * batch_size < tokens.shape[0]:
    num_batches += 1

batched_tokens = np.array_split(tokens, num_batches)
labels_dist = np.ones(shape=(len(common.countries)))

batch = 0
num_batches = len(batched_tokens)

text_labels = []

for tokens_in_this_batch in batched_tokens:
    print(f"💻 [{batch + 1}/{num_batches}] ...", end='\r')
    labels = model(tokens_in_this_batch, training=False).numpy()


    for i in range(labels.shape[0]):
        text_labels.append(labels[i])
        labels_dist += labels[i]
        labels_dist /= np.sum(labels_dist)

    batch += 1

print()

#labels_dist /= np.sum(labels_dist)

result = []

for i in range(labels_dist.shape[0]):
    country = common.countries[i]
    result.append((country, labels_dist[i]))


result = sorted(result, reverse=True, key=lambda n: n[1])

print("\n---🏴 Predicted native identification 🏴---")
for r in result:
    probability = int(r[1] * 10000) / 100.0

    if probability < 1.0:
        break

    print(f"\t{r[0]}: {probability}%")

print()

#
# Find the most significant comment
#

top_country = result[0][0]
top_country_idx = common.get_country_index(top_country)

top_score = 0.0
top_comment = ""

for i in range(len(text_labels)):
    label = text_labels[i]
    score = label[top_country_idx]

    if score > top_score:
        top_score = score
        top_comment = text[i]

probability = int(top_score * 10000) / 100.0

print(f"---📃 Highest scoring comment of {probability}% being from 🌍{top_country} ---")
print(top_comment)
print("---\n")

'''
print("--- Sentences scoring ---")

# Classify individual sentences

for i in range(labels.shape[0]):
    idx = labels[i].argmax()
    country = common.countries[idx]
    probability = int(labels[i][idx] * 10000) / 100.0

    l = len(text[i])
    l = min(60, l)
    brief = text[i][:l]

    print(f"[{country} {probability}%]\t\t{brief}...")

print()
'''