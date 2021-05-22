# %% IMPORTS
import os
import matplotlib.pyplot as plt

# tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no cuda warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# import tensorflow_datasets as tfds
# import tensorflow_hub as hub

# BERT
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

# appropriated class utils
from utils.process_features import extract_features, map_features, encode_data

# own utils
from utils.load_data import load_data

# %% CONSTANTS

#! argparse
BATCH_SIZE = 3 
EPOCHS = 3
OUTPATH = 'output'


# %% GET DATA

data_train, data_test = load_data(os.path.join('data', 'VideoCommentsThreatCorpus.txt'))

# %% SAMPLE DATA - REMOVE THIS PART

data_train = data_train[:100]
data_test = data_test[:100]

# %% ENCODE DATA

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# train dataset
data_train_encoded = encode_data(data_train, tokenizer).batch(BATCH_SIZE)
# test dataset
data_test_encoded = encode_data(data_test, tokenizer).batch(BATCH_SIZE)

# %% MODEL

# recommended learning rate for Adam 5e-5, 3e-5, 2e-5
learning_rate = 2e-5

# model initialization
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# optimizer Adam recommended
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, 
                                     epsilon=1e-08)

# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

# compile the model
model.compile(optimizer=optimizer, 
              loss=loss, 
              metrics=[metric])


# %% TRAIN
history = model.fit(data_train_encoded, 
                         epochs=EPOCHS,
                         batch_size=BATCH_SIZE,
                         validation_data=data_test_encoded)

# %% EVAL
loss, accuracy = model.evaluate(data_test_encoded)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')


# %% GRAPH
history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.savefig(os.path.join(OUTPATH, 'history.png'), bbox_inches='tight')
# %% SAVE HISTORY


#* Graphing, saving models etc
# https://www.tensorflow.org/tutorials/text/classify_text_with_bert
# %%
