# import numpy
# import tensorflow
# import webookcare.models
# from webookcare.tools.postprocess import PostProcess
# from webookcare.tools.preprocess import Preprocess
# from webookcare.layers import helper_layers
# from webookcare.tools import save_models
# from tensorflow.keras.backend import clear_session
# import gc
import pathlib
import os
from webookcare.tools.NLP.misc import (
    train, 
    predict, 
    determine_shapes
    )

current_dir = pathlib.Path(__file__).resolve().parent

DATA_FILE_PATH = "paraphrased_sentences.npy"
LABELS_FILE_PATH = "augmented_labels.npy"

VOCAB_PATH_RES = os.path.join(
    current_dir, 
    "train_data_int_1g_vocab.npy"
    ).replace('\\', '/')
LABELS_VOCAB_PATH_RES = os.path.join(
    current_dir, 
    "mh_labels_vocab.npy"
    ).replace('\\', '/')
DEFAULT_MODEL_PATH = os.path.join(
    current_dir, 
    "saved_models/MLP_int_1g_unigram_multihot_labels"
    ).replace('\\', '/')

NUM_NGRAMS = 1

# # num of columns needed for the vectorized input string (num of columns for vectorized training data)
# NUM_UNIQUE_ITEMS = 415
# # num of columns corresponding for multi-hot encoded labels
# NUM_LABELS_VOCAB_ITEMS = 12

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
          save_files=save_files)
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
    NUM_UNIQUE_ITEMS = 415
    NUM_LABELS_VOCAB_ITEMS = 12
    predictions = predict(data,
                          num_unique_items=NUM_UNIQUE_ITEMS,
                          vocab_path_res=VOCAB_PATH_RES,
                          labels_vocab_path_res=LABELS_VOCAB_PATH_RES,
                          default_model_path=DEFAULT_MODEL_PATH)
    return predictions


# def train(parent_dir,
#           data_file_path="paraphrased_sentences.npy", 
#           labels_file_path="augmented_labels.npy", 
#           epochs=10, 
#           batch_size=32, 
#           model_path=None,
#           save_files=True):
#     """
#     Trains a model using the input data and labels. The input data is processed, vectorized, and padded 
#     (if necessary), and then fed into the model to generate predictions. The model is trained using the 
#     processed data and labels, and the trained model is saved to the specified path.

#     Parameters
#     ----------
#     data : list or array-like
#         The input data to be trained on, typically a list of strings or sequences.
#     labels : list or array-like
#         The labels corresponding to the input data, typically a list of strings or sequences.
#     epochs : int, optional
#         The number of epochs to train the model for, by default 10.
#     batch_size : int, optional
#         The batch size to use during training, by default 32.
#     model_path : str, optional
#         The path to save the trained model to, by default None.

#     Returns
#     -------
#     str
#         The path to the saved model.
#     """
#     data_file_path = parent_dir.joinpath(data_file_path)
#     data = numpy.load(data_file_path).flatten().tolist()
#     labels_file_path = parent_dir.joinpath(labels_file_path)
#     labels = numpy.load(labels_file_path, 
#                         allow_pickle=True).flatten().tolist()
    
#     preo = Preprocess()
#     pp = PostProcess()
    
#     train_labels_od = preo.remove_special_chars(labels)

#     train_data_od, train_labels_od = pp.remove_nones(
#         data, 
#         train_labels_od
#     )

#     train_data_od = preo.clean_non_utf8(train_data_od)
#     train_labels_od = preo.clean_non_utf8(train_labels_od)
    
#     train_labels_od = [[' '.join(i for i in j)] for j in train_labels_od]
    
#     (train_labels_od_ng_vocab,
#      train_labels_od_ngrams) = pp.create_ngrams(train_labels_od, 
#                                                 NUM_NGRAMS)

#     (train_labels_od_ngrams_padded,
#      train_labels_od_ng_vocab_padded) = pp.pad_ngrams(
#         train_labels_od_ngrams,
#         train_labels_od_ng_vocab
#     )
#     (train_labels_od_ngrams_padded_mh,
#      train_labels_od_ngrams_padded_mh_vocab) = pp.multihot_encode(
#         train_labels_od_ngrams_padded,
#         train_labels_od_ng_vocab_padded
#     )
    
#     # NUM_UNIQUE_TERMS = train_labels_od_ngrams_padded_mh.shape[-1]

#     # converting input to tensors, vectorizing data and padding them if necessary
#     # vocab = numpy.load(VOCAB_PATH_RES)
#     # labels_vocab = numpy.load(LABELS_VOCAB_PATH_RES)
#     # labels = tensorflow.constant(labels)
#     # o = helper_layers.ProcessData(
#     #     training=True,
#     #     tvec_om='int', 
#     #     vocab=vocab,
#     #     pad_to_max_tokens=True, 
#     #     output_sequence_length=NUM_UNIQUE_ITEMS)
#     o = helper_layers.ProcessData(train_data_od, 20000, 'int')
#     train_data_od_t = tensorflow.constant(train_data_od) # tensor of strings
#     train_data_od_t = o.tvec(train_data_od_t)
    
#     if save_files:
#         numpy.save(parent_dir.joinpath("mh_labels_vocab.npy"),
#                    train_labels_od_ngrams_padded_mh_vocab)
#         train_data_int_1g_vocab_path = \
#             parent_dir.joinpath(pathlib.Path('train_data_int_1g_vocab'))
#         numpy.save(train_data_int_1g_vocab_path,
#             o.tvec.get_vocabulary())
    
#     model = webookcare.models.MLP('MLP', 
#                               dense_units=[
#                                   8192, 
#                                   4096, 
#                                   train_labels_od_ngrams_padded_mh.shape[-1],
#                                   ]
#                               )
#     model.compile()
#     history = model.fit(x=train_data_od_t,
#                         y=train_labels_od_ngrams_padded_mh,
#                         validation_split=0.3,
#                         epochs=10)
    
#     # data_txt = o.tvec(data)
#     # labels_txt = o.tvec(labels)
    
#     # loading the model
#     # model = save_models.load(DEFAULT_MODEL_PATH)
    
#     # training the model
#     # model.fit(data_txt, labels_txt, epochs=epochs, batch_size=batch_size)
    
#     # saving the model
#     if model_path is None:
#         model_path = DEFAULT_MODEL_PATH
#     save_models.save(model,
#                      'MLP_int_1g_unigram_multihot_labels',
#                      'default')
    
#     return model_path


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
#     vocab = numpy.load(VOCAB_PATH_RES)
#     labels_vocab = numpy.load(LABELS_VOCAB_PATH_RES)
    
#     # converting input to tensors, vectorizing data and padding them if necessary
#     data = tensorflow.constant(data)
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
#     # model = load_model()
#     model = save_models.load(DEFAULT_MODEL_PATH)
    
#     # predicting labels
#     if "serving_default" in model.signatures:
#         infer = model.signatures['serving_default']
#         predictions = infer(prediction_txt)
#         output_key = list(predictions.keys())[0]
#         predictions = predictions[output_key]
        
#     else:
#         predictions = model.predict(prediction_txt)
    
#     predictions = predictions[0]    
    
#     # if no labels are generated following steps would do more iterations forcing model to generate some
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
    