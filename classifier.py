# %% IMPORTS
import os

# tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no cuda warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_datasets as tfds
# import tensorflow_hub as hub

# BERT
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

# Own utils
from utils.load_data import load_data, reclassify_labels


# %% GET DATA

ds_train, ds_test = load_data(os.path.join('data', 'VideoCommentsThreatCorpus.txt'))
ds_train, ds_test = reclassify_labels(ds_train, ds_test)

# %% SAMPLE DATA - REMOVE THIS PART
ds_train = ds_train[:80]
ds_test = ds_test[:20]

print(ds_train)
print(ds_test)
#* TO HERE WORKS --------------------------------

# %% HELPERS

def extract_features(text): 
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        #! udregn en max length til at være den længste sætning, men højst 512 tokens
        max_length=160, # truncates if len(s) > max_length
        pad_to_max_length=True # pads to the right by default
    )

# map to the expected input to TFBertForSequenceClassification
def map_features(input_ids, attention_masks, token_type_ids, label): #! Reuse this function, remove 'example' from name
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks
      }, label

def encode_data(dataset): 
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    for review, label in tfds.as_numpy(dataset): #! no .as_numpy needed for our data
        bert_input = extract_features(review.decode()) #! review.decode() returns the plaintext (that we can extract from our own data directly)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_features)

# %% ENCODE DATA

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# train dataset
ds_train_encoded = encode_data(ds_train).shuffle(10000).batch(32)
# test dataset
ds_test_encoded = encode_data(ds_test).batch(32)

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
bert_history = model.fit(ds_train_encoded, 
                         epochs=number_of_epochs,
                         batch_size=32,
                         validation_data=ds_test_encoded)

# %% EVAL
loss, accuracy = model.evaluate(ds_test_encoded)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')
# %%
