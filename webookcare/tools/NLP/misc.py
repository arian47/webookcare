from webookcare.tools.postprocess import PostProcess
from webookcare.tools.preprocess import Preprocess
from tensorflow.keras.backend import clear_session
from webookcare.layers import helper_layers
from webookcare.tools import save_models
import numpy
import pathlib
import tensorflow
import webookcare.models

# TODO: need to differentiate between the way determine_shapes works 
# on the new and seen data
# def determine_shapes(parent_dir,
#                      save_req=False,
#                      data_file_path="paraphrased_sentences.npy", 
#                      labels_file_path="augmented_labels.npy",
#                      num_ngrams=1):
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
#     data_file_path=parent_dir.joinpath(data_file_path)
#     data = numpy.load(data_file_path).flatten().tolist()
#     # labels that were augmented and saved before 
#     labels_file_path=parent_dir.joinpath(labels_file_path)
#     labels = numpy.load(labels_file_path, 
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

#     # TODO: below line might not be true in all the cases
#     train_labels_od = [[' '.join(i for i in j)] for j in train_labels_od]
    
#     # creating n-grams for labels and vocab
#     (train_labels_od_ng_vocab,
#      train_labels_od_ngrams) = pp.create_ngrams(
#          train_labels_od, 
#          num_ngrams
#          )

#     # padding n-grams labels, vocab including <pad> keyword as well
#     (train_labels_od_ngrams_padded,
#      train_labels_od_ng_vocab_padded) = pp.pad_ngrams(
#          train_labels_od_ngrams,
#          train_labels_od_ng_vocab
#         )
    
#     # multihot encoded n-grams labels, adding <unk> keyword to the vocab that included <pad> keyword
#     (train_labels_od_ngrams_padded_mh,
#      train_labels_od_ngrams_padded_mh_vocab) = pp.multihot_encode(
#          train_labels_od_ngrams_padded,
#          train_labels_od_ng_vocab_padded
#          )
#     assert len(train_data_od) == len(train_labels_od)
    
#     # creating helper layer object needed for vectorizing
#     # shapes and saving
#     o = helper_layers.ProcessData(training=False,
#                                   input_data=train_data_od, 
#                                   tvec_mt=20000, 
#                                   tvec_om='int')
    
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


def train(parent_dir,
          training=True,
          data_file_path="paraphrased_sentences.npy", 
          labels_file_path="augmented_labels.npy", 
          epochs=10, 
          batch_size=32, 
          model_path=None,
          save_files=True,
          num_ngrams=1):
    """
    Trains a model using the input data and labels. The input data is processed, vectorized, and padded 
    (if necessary), and then fed into the model to generate predictions. The model is trained using the 
    processed data and labels, and the trained model is saved to the specified path.

    Parameters
    ----------
    data : list or array-like
        The input data to be trained on, typically a list of strings or sequences.
    labels : list or array-like
        The labels corresponding to the input data, typically a list of strings or sequences.
    epochs : int, optional
        The number of epochs to train the model for, by default 10.
    batch_size : int, optional
        The batch size to use during training, by default 32.
    model_path : str, optional
        The path to save the trained model to, by default None.

    Returns
    -------
    str
        The path to the saved model.
    """
    data_file_path = parent_dir.joinpath(data_file_path)
    data = numpy.load(data_file_path).flatten().tolist()
    labels_file_path = parent_dir.joinpath(labels_file_path)
    labels = numpy.load(labels_file_path, 
                        allow_pickle=True).flatten().tolist()
    
    preo = Preprocess()
    pp = PostProcess()
    
    train_labels_od = preo.remove_special_chars(labels)

    train_data_od, train_labels_od = pp.remove_nones(
        data, 
        train_labels_od
    )

    # same number of train data and labels
    assert len(train_data_od) == len(train_labels_od)

    train_data_od = preo.clean_non_utf8(train_data_od)
    train_labels_od = preo.clean_non_utf8(train_labels_od)
    
    # TODO: below line might not be true in all the cases
    train_labels_od = [[' '.join(i for i in j)] for j in train_labels_od]
    
    (train_labels_od_ng_vocab,
     train_labels_od_ngrams) = pp.create_ngrams(
         train_labels_od, 
         num_ngrams
         )

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
    
    # NUM_UNIQUE_TERMS = train_labels_od_ngrams_padded_mh.shape[-1]

    # converting input to tensors, vectorizing data and padding them if necessary
    # vocab = numpy.load(VOCAB_PATH_RES)
    # labels_vocab = numpy.load(LABELS_VOCAB_PATH_RES)
    # labels = tensorflow.constant(labels)
    # o = helper_layers.ProcessData(
    #     training=True,
    #     tvec_om='int', 
    #     vocab=vocab,
    #     pad_to_max_tokens=True, 
    #     output_sequence_length=NUM_UNIQUE_ITEMS)
    
    # tensor of strings
    train_data_od_t = tensorflow.constant(train_data_od)
    if training:
        o = helper_layers.ProcessData(training=True,
                                      input_data=train_data_od, 
                                      tvec_mt=20000, 
                                      tvec_om='int')
    else:
        o = helper_layers.ProcessData(training=False,
                                      input_data=train_data_od, 
                                      tvec_mt=20000, 
                                      tvec_om='int')
     
    train_data_od_t = o.tvec(train_data_od_t)
    
    if save_files:
        numpy.save(parent_dir.joinpath("mh_labels_vocab.npy"),
                   train_labels_od_ngrams_padded_mh_vocab)
        train_data_int_1g_vocab_path = \
            parent_dir.joinpath(pathlib.Path(f'train_data_int_{num_ngrams}g_vocab'))
        numpy.save(train_data_int_1g_vocab_path,
            o.tvec.get_vocabulary())
    
    if training:
        model = webookcare.models.MLP('MLP', 
                                      dense_units=[
                                          8192, 
                                          4096, 
                                          train_labels_od_ngrams_padded_mh.shape[-1],
                                          ]
                                      )
        model.compile()
        history = model.fit(x=train_data_od_t,
                            y=train_labels_od_ngrams_padded_mh,
                            validation_split=0.3,
                            epochs=epochs,
                            batch_size=batch_size)
        
        # data_txt = o.tvec(data)
        # labels_txt = o.tvec(labels)
        
        # loading the model
        # model = save_models.load(DEFAULT_MODEL_PATH)
        
        # training the model
        # model.fit(data_txt, labels_txt, epochs=epochs, batch_size=batch_size)
        
        # saving the model
        save_models.save(model,
                         f'MLP_int_{num_ngrams}g_unigram_multihot_labels',
                         'default')
        
        return model_path
    else:
        NUM_LABELS_VOCAB_ITEMS = train_labels_od_ngrams_padded_mh.shape[-1]
        NUM_UNIQUE_ITEMS = train_data_od_t.shape[-1]
        return NUM_UNIQUE_ITEMS, NUM_LABELS_VOCAB_ITEMS


def predict(data,
            num_unique_items,
            vocab_path_res,
            labels_vocab_path_res,
            default_model_path):
    """
    Predicts labels for a given input data using a pre-trained model. The input is processed,
    vectorized, and padded (if necessary), and then fed into the model to generate predictions.

    The model output is post-processed by applying a thresholding mechanism to classify the predicted 
    labels, which are then returned. The function will attempt to generate predictions for up to 10 
    iterations if the initial prediction yields no labels.

    Parameters
    ----------
    data : list or array-like
        The input data to be predicted, typically a list of strings or sequences.

    Returns
    -------
    list
        A list of predicted labels corresponding to the input data.
    """
    vocab = numpy.load(vocab_path_res)
    labels_vocab = numpy.load(labels_vocab_path_res)
    
    # converting input to tensors, vectorizing data and padding them if necessary
    data = tensorflow.constant(data)
    o = helper_layers.ProcessData(
        training=False,
        tvec_om='int', 
        vocab=vocab,
        pad_to_max_tokens=True, 
        output_sequence_length=num_unique_items)
    
    prediction_txt = o.tvec(data)
    
    # determining input shape to expand if necessary
    if prediction_txt.ndim < 2:
        prediction_txt = tensorflow.expand_dims(prediction_txt, 0)
    
    # loading the model
    # model = load_model()
    model = save_models.load(default_model_path)
    
    # predicting labels
    if "serving_default" in model.signatures:
        infer = model.signatures['serving_default']
        predictions = infer(prediction_txt)
        output_key = list(predictions.keys())[0]
        predictions = predictions[output_key]
        
    else:
        predictions = model.predict(prediction_txt)
    
    predictions = predictions[0]    
    
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