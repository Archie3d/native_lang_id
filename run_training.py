import os
import sys

# Use async GPU memory allocator
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import common
import config
import model
from dataset import TextSequence
import tensorflow as tf


BATCH_SIZE = 256
LEARNING_RATE = 1e-3

vocab = common.load_vocabulary("./dataset/vocab.txt")
tokenizer = common.make_bert_tokenizer(vocab, lower_case=True)

model = model.make_model(
    vocabulary_size=len(vocab),
    embedding_size=config.model['embedding_size'],
    lstm_size=config.model['lstm_size'],
    hidden_size=config.model['hidden_size']
)

# Loading weights of pretrained model
if len(sys.argv) > 1:
    checkpoint = sys.argv[1]
    if checkpoint.endswith('.index'):
        checkpoint = checkpoint[:len(checkpoint) - len('.index')]
    print(f"Continue training from checkpoint {checkpoint}")
    model.load_weights(checkpoint)

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

train_data = TextSequence(
    path="./dataset/train",
    countries=common.countries,
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer,
    shuffle=True,
    limit=10000,
    epsilon=0.002 # Apply classifier probabilities smoothing to mitigate overfitting
)

validation_data = TextSequence(
    path="./dataset/valid",
    countries=common.countries,
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer,
    shuffle=False,
    epsilon=0.0
)

#
#   TRAINING
#

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=100),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='lang_' + '{epoch:03d}-{val_loss:.8f}',
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        update_freq='epoch',
        histogram_freq=1,
        write_images=False
    )
]

model.fit(
    train_data,
    validation_data=validation_data,
    epochs=1000,
    callbacks=callbacks
)
