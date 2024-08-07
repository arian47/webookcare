import process_data
import pandas
import numpy
import models
import helper_layers
import tensorflow
import pathlib
import logging
import argparse

numpy.set_printoptions(threshold=numpy.inf)
# Set TensorFlow logging level to ERROR
tensorflow.get_logger().setLevel('ERROR')
# Suppress warnings from the Python logging module
logging.getLogger('tensorflow').setLevel(logging.ERROR)

DEFAULT_MODEL_PATH = './saved_model/MLP_int_1g_bigram_multihot_labels'


def load_model(saved_model_path):
    model = tensorflow.saved_model.load(saved_model_path)
    return model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Get predictions on single or multiple strings.')
    # parser.add_argument('--train', action='store_true', help='Train the model')
    # parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('-p', '--pred', type=list[str], help='prediction on the list of strings.')
    parser.add_argument('-l', '--load', type=str, default=DEFAULT_MODEL_PATH, 
                        help='Path to load the model')

    return parser.parse_args()


def main():
    pp = process_data.PostProcess()
    preo = process_data.Preprocess()
    # NUM_UNIQUE_ITEMS = train_data_od_t.shape[-1]
    NUM_UNIQUE_ITEMS = 710
    NUM_LABELS_VOCAB_ITEMS = 2531
    vocab_path = 'train_data_int_1g_vocab.npy'
    vocab = numpy.load(vocab_path)
    labels_vocab_path = 'mh_labels_vocab.npy'
    labels_vocab = numpy.load(labels_vocab_path)
    
    args = parse_arguments()
    assert args.pred, 'no item passed for prediction'
    # either pad with following option or manually below
    o = helper_layers.VectorizePrediction(args.pred, tvec_om='int', vocab=vocab,
                                          pad_to_max_tokens=True, output_sequence_length=710)
    if args.pred.shape[0] == 1:
        # single tensor
        data = tensorflow.expand_dims(tensorflow.constant(args.pred),
                                      0)
        data = pp.remove_nones(
          data
        )
        data = preo.clean_non_utf8(data)
    elif args.pred.shape[0] > 1:
        data = pp.remove_nones(
          data
        )
        data = preo.clean_non_utf8(data)
    data = o.tvec(data)
        
    if args.load:
        model = load_model(args.load)
        print(f'Model loaded from {args.load}')
    # else:
        # print('Please specify --train or --load')
        
    raw_predictions = model.predict(data)
    predictions = tensorflow.where(raw_predictions >= 0.5, 1, 0)
    predicted_labels = []
    for pred in predictions:
        indices = tensorflow.where(pred == 1).numpy().flatten()
        predicted_ngrams = [labels_vocab[idx] for idx in indices]
        predicted_labels.append(predicted_ngrams)
    print(predicted_labels)

if __name__ == '__main__':
    main()
    
    