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