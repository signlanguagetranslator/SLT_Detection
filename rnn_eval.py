"""
Run a holdout set of data through our trained RNN. Requires we first
run train_rnn.py and save the weights.
"""
from rnn_utils import get_network_wide, get_data
import argparse
import tensorflow as tf
import tflearn
import numpy as np
import sys


def load_labels(label_file):
    label = {}
    count = 0
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label[l.strip()] = count
        count += 1
    return label


def main(filename, frames, batch_size, num_classes, input_length):
    # Get our data.
    X, Y = get_data(input_data_dump, num_frames_per_video, labels, False)

    num_classes = len(labels)
    size_of_each_frame = X.shape[2]

    # Get our network.
    net = get_network_wide(num_frames_per_video, size_of_each_frame, num_classes)
    # Train the model.
    model = tflearn.DNN(net, tensorboard_verbose=0)
    try:
        model.load('checkpoints/' + model_file)
        print("\nModel Exists! Loading it")
        print("Model Loaded")
    except Exception:
        print("\nNo previous checkpoints of %s exist" % (model_file))
        print("Exiting..")
        sys.exit()

    predictions = model.predict(X)
    predictions = np.array([np.argmax(pred) for pred in predictions])
    Y = np.array([np.argmax(each) for each in Y])

    # Writing predictions and gold labels to file
    rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
    print(rev_labels)
    with open("result.txt", "w") as f:
        f.write("gold, pred\n")
        for a, b in zip(Y, predictions):
            f.write("%s %s\n" % (rev_labels[a], rev_labels[b]))

    acc = 100 * np.sum(predictions == Y) / len(Y)
    print("Accuracy: ", acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the RNN model')
    parser.add_argument("input_file_dump", help="file containing intermediate representation of gestures from inception model.")
    parser.add_argument("model_file", help="Name of the model file to be used for prediction.")
    parser.add_argument("--label_file", help="path to label file generated by inception", default="retrained_labels.txt")
    parser.add_argument("--batch_size", help="batch Size", default=32)
    args = parser.parse_args()

    labels = load_labels(args.label_file)
    input_data_dump = args.input_file_dump
    num_frames_per_video = 100
    batch_size = args.batch_size
    model_file = args.model_file

    main(input_data_dump, num_frames_per_video, batch_size, labels, model_file)
