import tensorflow as tf

# utils from class have been appropriated for this assignment

# tokenize the raw text for BERT
def extract_features(text, tokenizer): 
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_token_type_ids=True,
        return_attention_mask=True, # ignore padding tokens
        max_length=512, # explicitly wants a number, this is the max token sequence for BERT
        pad_to_max_length=True, # fills token sequence with 0s up to 512 tokens
        truncation=True # slice sequence to a max of 512 tokens (this probable does not happen with our data)
    )

# create mapping between features of a sequence and corresponding label
def map_features(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks
      }, label

def encode_data(dataset, tokenizer): 
    # prepare lists that can be converted to a tensorflow dataset
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    for label, comment_line in dataset:
        # tokenize the raw text
        bert_input = extract_features(comment_line, tokenizer)
        # unpack features into the above data structure
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])

    # create the tensorflow dataset structure
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_features)