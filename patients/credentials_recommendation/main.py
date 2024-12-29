import numpy
import tensorflow
import pathlib
import os
import logging
import typer
import webookcare.models
import webookcare.postprocess
import webookcare.preprocess
import webookcare.helper_layers
import webookcare.save_models
from tensorflow.keras.backend import clear_session
import gc

# app = typer.Typer()

# vocab for training data
# VOCAB_PATH_RES = pathlib.Path("./train_data_int_1g_vocab.npy").absolute().as_posix()
# VOCAB_PATH_RES = os.path.abspath("./train_data_int_1g_vocab.npy")
current_dir = pathlib.Path(__file__).resolve().parent
VOCAB_PATH_RES = os.path.join(
    current_dir, 
    "train_data_int_1g_vocab.npy").replace('\\', '/')
# print(VOCAB_PATH_RES)
# VOCAB_PATH_RES = repr(VOCAB_PATH_RES)[1:-1]
# VOCAB_PATH_RES = pathlib.Path("train_data_int_1g_vocab.npy")
# assert VOCAB_PATH_RES.exists(), f'{VOCAB_PATH_RES} file doesn\'t exist!'
# vocab for multi-hot encoded labels
# LABELS_VOCAB_PATH_RES = pathlib.Path('./mh_labels_vocab.npy').absolute().as_posix()
# LABELS_VOCAB_PATH_RES = repr(LABELS_VOCAB_PATH_RES)[1:-1]
# LABELS_VOCAB_PATH_RES = os.path.abspath('./mh_labels_vocab.npy')
LABELS_VOCAB_PATH_RES = os.path.join(
    current_dir, 
    "mh_labels_vocab.npy").replace('\\', '/')
# LABELS_VOCAB_PATH_RES = pathlib.Path('mh_labels_vocab.npy')
# assert LABELS_VOCAB_PATH_RES.exists(), f'{LABELS_VOCAB_PATH_RES} file doesn\'t exist!'
DEFAULT_MODEL_PATH = os.path.join(
    current_dir, 
    "saved_models/MLP_int_1g_unigram_multihot_labels").replace('\\', '/')



# num of columns needed for the vectorized input string (num of columns for vectorized training data)
NUM_UNIQUE_ITEMS = 415
# num of columns corresponding for multi-hot encoded labels
NUM_LABELS_VOCAB_ITEMS = 12

# @app
def determine_shapes(save_req=False):
    # sentences that were paraphrased and saved before
    data = numpy.load("paraphrased_sentences.npy").flatten().tolist()
    # labels that were augmented and saved before 
    labels = numpy.load("augmented_labels.npy", 
                        allow_pickle=True).flatten().tolist()
    # initializing pre and post processing objs
    preo = webookcare.preprocess.Preprocess()
    pp = webookcare.postprocess.PostProcess()
    # removing special character, nones, non-utf8 chars, 
    train_labels_od = preo.remove_special_chars(labels)

    train_data_od, train_labels_od = pp.remove_nones(
        data, 
        train_labels_od
    )
    # same number of train data and labels
    assert len(train_data_od) == len(train_labels_od)
    
    train_data_od = preo.clean_non_utf8(train_data_od)
    train_labels_od = preo.clean_non_utf8(train_labels_od)

    train_labels_od = [[' '.join(i for i in j)] for j in train_labels_od]
    
    # creating bigrams for labels and vocab
    (train_labels_od_ng_vocab,
    train_labels_od_ngrams) = pp.create_ngrams(train_labels_od, 1)

    # padding bigram labels, vocab including <pad> keyword as well
    (train_labels_od_ngrams_padded,
    train_labels_od_ng_vocab_padded) = pp.pad_ngrams(
        train_labels_od_ngrams,
        train_labels_od_ng_vocab
    )
    
    # multihot encoded bigram labels, adding <unk> keyword to the vocab that included <pad> keyword
    (train_labels_od_ngrams_padded_mh,
    train_labels_od_ngrams_padded_mh_vocab) = pp.multihot_encode(
        train_labels_od_ngrams_padded,
        train_labels_od_ng_vocab_padded
    )
    assert len(train_data_od) == len(train_labels_od)

    print(len(train_data_od), len(train_labels_od))
    
    # creating helper layer object needed for vectorizing
    # shapes and saving
    o = webookcare.helper_layers.ProcessData(train_data_od, 
                                             20000, 
                                             'int')
    
    
    train_data_od_t = tensorflow.constant(train_data_od) # tensor of strings
    train_data_od_t = o.tvec(train_data_od_t)
    
    if save_req:
        # saving the vocabulary for labels
        numpy.save("mh_labels_vocab.npy", # multi-hot encoded labels vocabulary includes <pad> and <unk> tokens
                   train_labels_od_ngrams_padded_mh_vocab)
        
        # saving the vocabulary for train data
        train_data_int_1g_vocab_path = pathlib.Path('train_data_int_1g_vocab')
        numpy.save(train_data_int_1g_vocab_path,
                   o.tvec.get_vocabulary())
    NUM_LABELS_VOCAB_ITEMS = train_labels_od_ngrams_padded_mh.shape[-1]
    NUM_UNIQUE_ITEMS = train_data_od_t.shape[-1]
    return NUM_UNIQUE_ITEMS, NUM_LABELS_VOCAB_ITEMS

# @app
# def load_model(name='MLP_int_1g_unigram_multihot_labels'):
def load_model():
    try:
        model = webookcare.save_models.load(DEFAULT_MODEL_PATH)
    except tensorflow.errors.ResourceExhaustedError or \
        tensorflow.errors.InternalError:
            tensorflow.keras.backend.clear_session()
            gc.collect()
            model = webookcare.save_models.load(DEFAULT_MODEL_PATH)
    return model

# @app
def predict(data):
    vocab = numpy.load(VOCAB_PATH_RES)
    labels_vocab = numpy.load(LABELS_VOCAB_PATH_RES)
    
    # converting input to tensors, vectorizing data and padding them if necessary
    data = tensorflow.constant(data)
    o = webookcare.helper_layers.ProcessData(
        training=False,
        tvec_om='int', 
        vocab=vocab,
        pad_to_max_tokens=True, 
        output_sequence_length=NUM_UNIQUE_ITEMS)
    
    prediction_txt = o.tvec(data)
    
    # determining input shape to expand if necessary
    if prediction_txt.ndim < 2:
        prediction_txt = tensorflow.expand_dims(prediction_txt, 0)
    
    # loading the model
    model = load_model()
    # print(model.__class__)
    
    if "serving_default" in model.signatures:
        infer = model.signatures['serving_default']
        predictions = infer(prediction_txt)
        output_key = list(predictions.keys())[0]
        predictions = predictions[output_key]
        
    else:
        predictions = model.predict(prediction_txt)
    
    predictions = predictions[0]
    # predicting labels
    
    
    # if no labels are generated following steps would do more iterations forcing model to generate some
    STATE = True
    predictions = tensorflow.cast(predictions, 
                                  dtype=tensorflow.float32)
    threshold = .5
    min_threshold = 0.0
    tolerance = 1e-6
    max_iterations = 10  # Limit the number of iterations
    iterations = 0
    while STATE and iterations<max_iterations:
        predictions = \
            tensorflow.where(tensorflow.cast(predictions,
                                             tensorflow.float32)>=threshold, 1, 0)
        predicted_labels = []
        if predictions.ndim > 1:
            for pred in predictions:
                indices = tensorflow.where(pred == 1).numpy().flatten()
                predicted_ngrams = [labels_vocab[i] for i in indices]
                predicted_labels.extend(predicted_ngrams)
        else:
            indices = tensorflow.where(predictions == 1).numpy().flatten()
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
    return predicted_labels

# print(predict(['Need help']))

# if __name__ == '__main__':
#     numpy.set_printoptions(threshold=numpy.inf)
#     # Set TensorFlow logging level to ERROR
#     tensorflow.get_logger().setLevel('ERROR')
#     # Suppress warnings from the Python logging module
#     logging.getLogger('tensorflow').setLevel(logging.ERROR)
#     # app()
    
    