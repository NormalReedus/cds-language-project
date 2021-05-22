# %% IMPORTS
import os

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
from utils.load_data import load_data, reclassify_labels


# %% GET DATA

data_train, data_test = load_data(os.path.join('data', 'VideoCommentsThreatCorpus.txt'))
# data_train, data_test = reclassify_labels(data_train, data_test)

# %% SAMPLE DATA - REMOVE THIS PART

data_train = data_train[:100]
data_test = data_test[:20]

# %% ENCODE DATA

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# train dataset
data_train_encoded = encode_data(data_train, tokenizer).batch(64)
# test dataset
data_test_encoded = encode_data(data_test, tokenizer).batch(64)

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
number_of_epochs = 1
bert_history = model.fit(data_train_encoded, 
                         epochs=number_of_epochs,
                         batch_size=64,
                         validation_data=data_test_encoded)

# %% EVAL
loss, accuracy = model.evaluate(data_test_encoded)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
# %%
