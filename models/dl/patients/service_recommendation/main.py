import os
import numpy
import webookcare.models
import tensorflow
import pathlib
import webookcare.postprocess
import webookcare.preprocess
import webookcare.helper_layers


current_dir = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = os.path.join(
        current_dir, 
        "saved_models/MLP_int_1g_bigram_multihot_labels").replace('\\', '/')
VOCAB_PATH_RES = os.path.join(current_dir, 
                              "train_data_int_1g_vocab.npy").replace('\\', '/')
LABELS_VOCAB_PATH_RES = os.path.join(current_dir, 
                                     "mh_labels_vocab.npy").replace('\\', '/')

# num of columns needed for the vectorized input string
NUM_UNIQUE_ITEMS = 710
# num of columns corresponding to the multihot encoded labels ndarray second axis
NUM_LABELS_VOCAB_ITEMS = 2531

def load_model():
    model = tensorflow.saved_model.load(DEFAULT_MODEL_PATH)
    return model

def predict(data):
    # pp = webookcare.postprocess.PostProcess()
    # preo = webookcare.preprocess.Preprocess()
    vocab = numpy.load(VOCAB_PATH_RES)
    labels_vocab = numpy.load(LABELS_VOCAB_PATH_RES)
    # converting input to tensors, vectorizing data and padding them if necessary
    data = tensorflow.constant(data)
    # either pad with following option or manually below
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
    
    