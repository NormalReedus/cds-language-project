# load model from file
# take user input in loop and predict
# print prediction
import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # no cuda warnings
import argparse
# from pathlib import Path

def main(model_path):
    model = tf.saved_model.load(model_path)

    while True:
        res = input('Offend me, please\n\n>')
        model.predict(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "try out the hate speech classifier model")
   
    parser.add_argument("-m", "--model_path", default=os.path.join('output', 'hate_model'), type = str, help = "path to the saved classifier model")

    args = parser.parse_args()
    
    main(model_path = args.model_path)