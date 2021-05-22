import tensorflow as tf

# utils from class have been appropriated for this assignment

def extract_features(text, tokenizer): 
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        max_length=512, # explicitly wants a number, this is the max token sequence for BERT
        pad_to_max_length=True, # pads to the right by default
        truncation=True
    )

# map to the expected input to TFBertForSequenceClassification
def map_features(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks
      }, label

def encode_data(dataset, tokenizer): 
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []

    for label, comment_line in dataset:
        bert_input = extract_features(comment_line, tokenizer)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_features)