import pandas
import numpy
import models
import tensorflow
import pathlib
import logging
import argparse
import train
import download_res
from .. import postprocess
from .. import preprocess
from .. import helper_layers

numpy.set_printoptions(threshold=numpy.inf)
# Set TensorFlow logging level to ERROR
tensorflow.get_logger().setLevel('ERROR')
# Suppress warnings from the Python logging module
logging.getLogger('tensorflow').setLevel(logging.ERROR)


DEFAULT_MODEL_PATH = pathlib.Path('MLP_int_1g_bigram_multihot_labels').absolute().as_posix()
vocab_path_res = pathlib.Path("train_data_int_1g_vocab.npy").absolute().as_posix()
labels_vocab_path_res = pathlib.Path('mh_labels_vocab.npy').absolute().as_posix()

# num of columns needed for the vectorized input string
NUM_UNIQUE_ITEMS = 710
# num of columns corresponding to the multihot encoded labels ndarray second axis
NUM_LABELS_VOCAB_ITEMS = 2531

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

def load_model(saved_model_path):
    model = tensorflow.saved_model.load(saved_model_path)
    return model

def parse_arguments():
    """Create the CLI which accepts parameters needed for the prediction models
    to perform correctly.
    """
    
    DESCRIPTION = """\
        Prediction Model for Matching Care Seeker/Giver
        ------------------------------------------------
            Models employed:
                Service Recommender: multilabel classification based on multihot encoded 
                labels and vectorized care need from patients.
                """
    parser = argparse.ArgumentParser(prog='PredictionModel',
                                     description=DESCRIPTION,
                                     formatter_class=CustomFormatter,
                                     epilog=' ',
                                     exit_on_error=False)
    parser.add_argument('predict', type=str, nargs='+',
                        help='prediction on single strings.')
    parser.add_argument('-l', '--load', dest='saved_model', type=str, 
                        default=DEFAULT_MODEL_PATH, help='Path to load the model', 
                        nargs='?', required=False)
    try:
        res = parser.parse_args()
    except argparse.ArgumentError:
        print('Got an arguemntError')
    return res


def main():
    pp = postprocess.PostProcess()
    preo = preprocess.Preprocess()
    # if not all([
        # vocab_path_res.exists(),
        # labels_vocab_path_res.exists(),
        # DEFAULT_MODEL_PATH.exists(),
    # ]):
        # service = download_res.authenticate()
        # if not vocab_path_res.exists():
            # download_res.download_file(service, download_res.vocab_res_id, 
                                    #    vocab_path_res)
        # if not labels_vocab_path_res.exists():
            # download_res.download_file(service, download_res.labels_vocab_res_id, 
                                    #    labels_vocab_path_res)
        # if not DEFAULT_MODEL_PATH.exists():
            # download_res.download_folder(service, download_res.saved_model_res_id, 
                                        #  DEFAULT_MODEL_PATH)
    vocab = numpy.load(vocab_path_res)
    labels_vocab = numpy.load(labels_vocab_path_res)
    
    args = parse_arguments()
    assert args.predict, 'no string passed for prediction'
    data = tensorflow.constant(args.predict)
    # either pad with following option or manually below
    o = helper_layers.VectorizePrediction(data, tvec_om='int', vocab=vocab,
                                          pad_to_max_tokens=True, 
                                          output_sequence_length=NUM_UNIQUE_ITEMS)
        # data = pp.remove_nones(
        #   data
        # )
        # data = preo.clean_non_utf8(data)
    # elif data.shape[0] > 1:
        # data = pp.remove_nones(
        #   data
        # )
        # data = preo.clean_non_utf8(data)
    data = o.tvec(data)
    try:
        model = load_model(args.saved_model)
        print(f'Model loaded from {args.saved_model}')
    except:
        model = train.train_model()
        print(f'Model trained from scratch')
    infer = model.signatures['serving_default']
    
    raw_predictions = infer(data)
    output_key = list(raw_predictions.keys())[0]
    STATE = True
    threshold = .5
    min_threshold = 0.0
    tolerance = 1e-6
    max_iterations = 10  # Limit the number of iterations
    iterations = 0
    while STATE and iterations < max_iterations:
        predictions = tensorflow.where(raw_predictions[output_key] >= threshold, 1, 0)
        predicted_labels = []
        for pred in predictions:
            indices = tensorflow.where(pred == 1).numpy().flatten()
            predicted_ngrams = [labels_vocab[i] for i in indices]
            predicted_labels.extend(predicted_ngrams)
        predicted_labels = [i for i in predicted_labels if i != '<PAD>']
        if predicted_labels != []:
            STATE = False
        elif predicted_labels == []:
            threshold -= .1
            iterations += 1
            if threshold < min_threshold + tolerance:
                break
    
    print(predicted_labels)

if __name__ == '__main__':
    main()
    
    