import os
# import gc
# import numpy
# import tensorflow
import pathlib
# from webookcare.tools.postprocess import PostProcess
# from webookcare.tools.preprocess import Preprocess
# from webookcare.layers import helper_layers
# from webookcare.tools import save_models
from webookcare.tools.NLP.misc import (train, 
                                       predict, 
                                       determine_shapes)

DATA_FILE_PATH = "paraphrased_sentences.npy"
LABELS_FILE_PATH = "augmented_labels.npy"

current_dir = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = os.path.join(
        current_dir, 
        "saved_models/MLP_int_1g_bigram_multihot_labels"
        ).replace('\\', '/')
VOCAB_PATH_RES = os.path.join(
    current_dir, 
    "train_data_int_1g_vocab.npy"
    ).replace('\\', '/')
LABELS_VOCAB_PATH_RES = os.path.join(
    current_dir, 
    "mh_labels_vocab.npy"
    ).replace('\\', '/')

# # num of columns needed for the vectorized input string
# NUM_UNIQUE_ITEMS = 710
# # num of columns corresponding to the multihot encoded labels ndarray second axis
# NUM_LABELS_VOCAB_ITEMS = 2531


NUM_NGRAMS = 2

def train_model(epochs=10, 
                batch_size=32, 
                model_path=DEFAULT_MODEL_PATH, 
                save_files=True):
    print('Starting training...')
    train(parent_dir=current_dir,
          data_file_path=DATA_FILE_PATH,
          labels_file_path=LABELS_FILE_PATH,
          epochs=epochs,
          batch_size=batch_size,
          model_path=model_path,
          save_files=save_files,
          )
    print('Training complete.')

def predict_data(data):
    # TODO: need to create another function for this
    # (NUM_UNIQUE_ITEMS,
    #  NUM_LABELS_VOCAB_ITEMS) = determine_shapes(
        #  parent_dir=current_dir,
        #  save_req=False,
        #  data_file_path=DATA_FILE_PATH,
        #  labels_file_path=LABELS_FILE_PATH,
        #  num_ngrams=NUM_NGRAMS
    #  )
    # test case
    NUM_UNIQUE_ITEMS = 710
    NUM_LABELS_VOCAB_ITEMS = 2531
    predictions = predict(data,
                          num_unique_items=NUM_UNIQUE_ITEMS,
                          vocab_path_res=VOCAB_PATH_RES,
                          labels_vocab_path_res=LABELS_VOCAB_PATH_RES,
                          default_model_path=DEFAULT_MODEL_PATH)
    return predictions


# def determine_shapes(save_req=False):
#     """
#     Determines and returns the shapes of the train data and label vocabularies after 
#     processing steps such as cleaning, n-gram creation, padding, and multi-hot encoding.

#     This function performs the following steps:
#     1. Loads and preprocesses paraphrased sentences and augmented labels.
#     2. Cleans data and removes invalid entries (e.g., non-UTF8 characters, Nones).
#     3. Generates bigrams for the labels and vocabularies.
#     4. Pads and encodes the bigrams using multi-hot encoding.
#     5. Vectorizes the training data using a helper layer and returns the final shapes.

#     Parameters
#     ----------
#     save_req : bool, optional
#         If True, saves the vocabulary for labels and training data to disk. Default is False.

#     Returns
#     -------
#     tuple
#         A tuple containing the shapes of the unique items in training data and the label vocabulary items.
#     """
#     # sentences that were paraphrased and saved before
#     data = numpy.load("paraphrased_sentences.npy").flatten().tolist()
#     # labels that were augmented and saved before 
#     labels = numpy.load("augmented_labels.npy", 
#                         allow_pickle=True).flatten().tolist()
#     # initializing pre and post processing objs
#     preo = Preprocess()
#     pp = PostProcess()
#     # removing special character, nones, non-utf8 chars, 
#     train_labels_od = preo.remove_special_chars(labels)

#     train_data_od, train_labels_od = pp.remove_nones(
#         data, 
#         train_labels_od
#     )
#     # same number of train data and labels
#     assert len(train_data_od) == len(train_labels_od)
    
#     train_data_od = preo.clean_non_utf8(train_data_od)
#     train_labels_od = preo.clean_non_utf8(train_labels_od)

#     train_labels_od = [[' '.join(i for i in j)] for j in train_labels_od]
    
#     # creating bigrams for labels and vocab
#     (train_labels_od_ng_vocab,
#     train_labels_od_ngrams) = pp.create_ngrams(train_labels_od, 1)

#     # padding bigram labels, vocab including <pad> keyword as well
#     (train_labels_od_ngrams_padded,
#     train_labels_od_ng_vocab_padded) = pp.pad_ngrams(
#         train_labels_od_ngrams,
#         train_labels_od_ng_vocab
#     )
    
#     # multihot encoded bigram labels, adding <unk> keyword to the vocab that included <pad> keyword
#     (train_labels_od_ngrams_padded_mh,
#     train_labels_od_ngrams_padded_mh_vocab) = pp.multihot_encode(
#         train_labels_od_ngrams_padded,
#         train_labels_od_ng_vocab_padded
#     )
#     assert len(train_data_od) == len(train_labels_od)
    
#     # creating helper layer object needed for vectorizing
#     # shapes and saving
#     o = helper_layers.ProcessData(train_data_od, 
#                                   20000, 
#                                   'int')
    
    
#     train_data_od_t = tensorflow.constant(train_data_od) # tensor of strings
#     train_data_od_t = o.tvec(train_data_od_t)
    
#     if save_req:
#         # saving the vocabulary for labels
#         numpy.save("mh_labels_vocab.npy", # multi-hot encoded labels vocabulary includes <pad> and <unk> tokens
#                    train_labels_od_ngrams_padded_mh_vocab)
        
#         # saving the vocabulary for train data
#         train_data_int_1g_vocab_path = pathlib.Path('train_data_int_1g_vocab')
#         numpy.save(train_data_int_1g_vocab_path,
#                    o.tvec.get_vocabulary())
#     NUM_LABELS_VOCAB_ITEMS = train_labels_od_ngrams_padded_mh.shape[-1]
#     NUM_UNIQUE_ITEMS = train_data_od_t.shape[-1]
#     return NUM_UNIQUE_ITEMS, NUM_LABELS_VOCAB_ITEMS


# def load_model():
#     """
#     Loads a pre-trained model from disk. If loading fails due to resource exhaustion or internal error,
#     the session is cleared, garbage collection is performed, and the model is attempted to load again.

#     Returns
#     -------
#     tensorflow.keras.Model
#         The loaded Keras model.
#     """
#     try:
#         model = save_models.load(DEFAULT_MODEL_PATH)
#     except tensorflow.errors.ResourceExhaustedError or \
#         tensorflow.errors.InternalError:
#             tensorflow.keras.backend.clear_session()
#             gc.collect()
#             model = save_models.load(DEFAULT_MODEL_PATH)
#     return model

# def predict(data):
#     """
#     Predicts labels for a given input data using a pre-trained model. The input is processed,
#     vectorized, and padded (if necessary), and then fed into the model to generate predictions.

#     The model output is post-processed by applying a thresholding mechanism to classify the predicted 
#     labels, which are then returned. The function will attempt to generate predictions for up to 10 
#     iterations if the initial prediction yields no labels.

#     Parameters
#     ----------
#     data : list or array-like
#         The input data to be predicted, typically a list of strings or sequences.

#     Returns
#     -------
#     list
#         A list of predicted labels corresponding to the input data.
#     """
#     # pp = webookcare.postprocess.PostProcess()
#     # preo = webookcare.preprocess.Preprocess()
#     vocab = numpy.load(VOCAB_PATH_RES)
#     labels_vocab = numpy.load(LABELS_VOCAB_PATH_RES)
#     # converting input to tensors, vectorizing data and padding them if necessary
#     data = tensorflow.constant(data)
#     # either pad with following option or manually below
#     o = helper_layers.ProcessData(
#         training=False,
#         tvec_om='int', 
#         vocab=vocab,
#         pad_to_max_tokens=True, 
#         output_sequence_length=NUM_UNIQUE_ITEMS)
#     prediction_txt = o.tvec(data)
#     # determining input shape to expand if necessary
#     if prediction_txt.ndim < 2:
#         prediction_txt = tensorflow.expand_dims(prediction_txt, 0)
    
#     # loading the model
#     model = save_models.load(DEFAULT_MODEL_PATH)
#     # print(model.__class__)
    
#     if "serving_default" in model.signatures:
#         infer = model.signatures['serving_default']
#         predictions = infer(prediction_txt)
#         output_key = list(predictions.keys())[0]
#         predictions = predictions[output_key]
        
#     else:
#         predictions = model.predict(prediction_txt)
    
#     predictions = predictions[0]
    
#     STATE = True
#     predictions = tensorflow.cast(predictions, 
#                                   dtype=tensorflow.float32)
#     threshold = .5
#     min_threshold = 0.0
#     tolerance = 1e-6
#     max_iterations = 10  # Limit the number of iterations
#     iterations = 0
#     while STATE and iterations<max_iterations:
#         predictions = \
#             tensorflow.where(tensorflow.cast(predictions,
#                                              tensorflow.float32)>=threshold, 1, 0)
#         predicted_labels = []
#         if predictions.ndim > 1:
#             for pred in predictions:
#                 indices = tensorflow.where(pred == 1).numpy().flatten()
#                 predicted_ngrams = [labels_vocab[i] for i in indices]
#                 predicted_labels.extend(predicted_ngrams)
#         else:
#             indices = tensorflow.where(predictions == 1).numpy().flatten()
#             predicted_ngrams = [labels_vocab[i] for i in indices]
#             predicted_labels.extend(predicted_ngrams)
#         predicted_labels = [i for i in predicted_labels if i != '<PAD>']
#         if predicted_labels != []:
#             STATE = False
#         elif predicted_labels == []:
#             threshold -= .1
#             iterations += 1
#             if threshold < min_threshold + tolerance:
#                 break
#     return predicted_labels
    
    