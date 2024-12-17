import numpy
import tensorflow
import pathlib
import logging
import typer
import webookcare.models
import webookcare.postprocess
import webookcare.preprocess
import webookcare.helper_layers
import webookcare.save_models

from tensorflow.keras.backend import clear_session
import gc

app = typer.Typer()

numpy.set_printoptions(threshold=numpy.inf)
# Set TensorFlow logging level to ERROR
tensorflow.get_logger().setLevel('ERROR')
# Suppress warnings from the Python logging module
logging.getLogger('tensorflow').setLevel(logging.ERROR)


vocab_path_res = pathlib.Path("./train_data_int_1g_vocab.npy").absolute()
labels_vocab_path_res = pathlib.Path('./mh_labels_vocab.npy').absolute()

# num of columns needed for the vectorized input string
NUM_UNIQUE_ITEMS = 415
# num of columns corresponding to the multihot encoded labels ndarray second axis
NUM_LABELS_VOCAB_ITEMS = 12

@app
def determine_shapes(save_req=False):
    data = numpy.load("paraphrased_sentences.npy").flatten().tolist()
    labels = numpy.load("augmented_labels.npy", 
                        allow_pickle=True).flatten().tolist()
    preo = webookcare.preprocess.Preprocess()
    pp = webookcare.postprocess.PostProcess()
    train_labels_od = preo.remove_special_chars(labels)

    train_data_od, train_labels_od = pp.remove_nones(
        data, 
        train_labels_od
    )

    train_data_od = preo.clean_non_utf8(train_data_od)
    train_labels_od = preo.clean_non_utf8(train_labels_od)

    train_labels_od = [[' '.join(i for i in j)] for j in train_labels_od]
    
    (train_labels_od_ng_vocab,
    train_labels_od_ngrams) = pp.create_ngrams(train_labels_od, 1)

    (train_labels_od_ngrams_padded,
    train_labels_od_ng_vocab_padded) = pp.pad_ngrams(
        train_labels_od_ngrams,
        train_labels_od_ng_vocab
    )
    (train_labels_od_ngrams_padded_mh,
    train_labels_od_ngrams_padded_mh_vocab) = pp.multihot_encode(
        train_labels_od_ngrams_padded,
        train_labels_od_ng_vocab_padded
    )
    assert len(train_data_od) == len(train_labels_od)

    print(len(train_data_od), len(train_labels_od))
    
    # shapes and saving
    o = webookcare.helper_layers.ProcessData(train_data_od, 
                                             20000, 
                                             'int')
    train_data_od_t = tensorflow.constant(train_data_od) # tensor of strings
    train_data_od_t = o.tvec(train_data_od_t)
    
    if save_req:
        numpy.save("mh_labels_vocab.npy",
                   train_labels_od_ngrams_padded_mh_vocab)
        train_data_int_1g_vocab_path = pathlib.Path('train_data_int_1g_vocab')
        numpy.save(train_data_int_1g_vocab_path,
                   o.tvec.get_vocabulary())
    NUM_LABELS_VOCAB_ITEMS = train_labels_od_ngrams_padded_mh.shape[-1]
    NUM_UNIQUE_ITEMS = train_data_od_t.shape[-1]
    return NUM_UNIQUE_ITEMS, NUM_LABELS_VOCAB_ITEMS

@app
def load_model(name='MLP_int_1g_unigram_multihot_labels'):
    try:
        model = webookcare.save_models.load(name)
    except tensorflow.errors.ResourceExhaustedError or \
        tensorflow.errors.InternalError:
            tensorflow.keras.backend.clear_session()
            gc.collect()
            model = webookcare.save_models.load(name)
    return model

@app
def predict(data):
    vocab = numpy.load(vocab_path_res)
    labels_vocab = numpy.load(labels_vocab_path_res)
    data = tensorflow.constant(data)
    o = webookcare.helper_layers.VectorizePrediction(data, 
                                                     tvec_om='int', 
                                                     vocab=vocab,
                                                     pad_to_max_tokens=True, 
                                                     output_sequence_length=NUM_UNIQUE_ITEMS)
    
    prediction_txt = o.tvec(data)
    model = load_model()
    if prediction_txt.ndim < 2:
        prediction_txt = tensorflow.expand_dims(prediction_txt, 0)
    predictions = model.predict(prediction_txt)
    STATE = True
    predictions = predictions[0]
    predictions = tensorflow.cast(predictions, 
                                  dtype=tensorflow.float32)
    threshold = .5
    min_threshold = 0.0
    tolerance = 1e-6
    max_iterations = 10  # Limit the number of iterations
    iterations = 0
    while STATE and iterations < max_iterations:
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

if __name__ == '__main__':
    app()
    
    