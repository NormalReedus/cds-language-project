# %% IMPORTS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no cuda warnings
import argparse

import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification

# appropriated class utils
from utils.process_features import extract_features, map_features, encode_data

# own utils
from utils.load_data import load_data


def main(sample_num, batch_size, epochs):
    outpath = 'output'

    data_train, data_test = load_data(os.path.join('data', 'VideoCommentsThreatCorpus.txt'), sample_num = sample_num)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # using the tokenizer to format our data as BERT wants it
    data_train_encoded = encode_data(data_train, tokenizer).batch(batch_size)
    data_test_encoded = encode_data(data_test, tokenizer).batch(batch_size)


    # define model and params
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

    # recommended learning rate for Adam 5e-5, 3e-5, 2e-5
    learning_rate = 2e-5

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

    # train the model
    history = model.fit(data_train_encoded, 
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=data_test_encoded)
    
    # save the model to ./output
    model.save(os.path.join(outpath, 'hate_model'), include_optimizer=False)

    # print validation loss / accuracy
    loss, accuracy = model.evaluate(data_test_encoded)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    # save history graph (taken from BERT docs)
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
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.savefig(os.path.join(outpath, 'history.png'), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "train the hate speech classifier")
   
    parser.add_argument("-s", "--sample_num", type = int, help = "whether to train on a subset of data points and how many to use")
    parser.add_argument("-b", "--batch_size", default = 32, type = int, help = "how many data points to train on at a time")
    parser.add_argument("-e", "--epochs", default = 3, type = int, help = "the number of epochs to train the model")

    args = parser.parse_args()
    
    main(sample_num = args.sample_num, batch_size = args.batch_size, epochs = args.epochs)