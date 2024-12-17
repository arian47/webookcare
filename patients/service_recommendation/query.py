import numpy
import requests
import json
import logging
import argparse
import pathlib
import tensorflow
import sys
import os
import webookcare.helper_layers
import webookcare.postprocess
import webookcare.preprocess
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# sys.path.insert(0, parent_dir)
# print(sys.path)



class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

class Query:
    # num of columns needed for the vectorized input string
    NUM_UNIQUE_ITEMS = 710
    # num of columns corresponding to the multihot encoded labels ndarray second axis
    NUM_LABELS_VOCAB_ITEMS = 2531
    def __init__(self, model_name:str='careseeker_task_cls', port_num:int=8501, 
                 end_point:str='predict'):
        self.url = f"http://142.93.159.195:{port_num}/v1/models/{model_name}:{end_point}"
        self.vocab_path_res = pathlib.Path("train_data_int_1g_vocab.npy").absolute().as_posix()
        self.labels_vocab_path_res = pathlib.Path("mh_labels_vocab.npy").absolute().as_posix()
    
    def prepare_data(self, data):
        pp = webookcare.postprocess.PostProcess()
        preo = webookcare.preprocess.Preprocess()
        self.vocab = numpy.load(self.vocab_path_res)
        self.labels_vocab = numpy.load(self.labels_vocab_path_res)
        o = webookcare.helper_layers.VectorizePrediction(data, tvec_om='int', vocab=self.vocab,
                                                         pad_to_max_tokens=True, 
                                                         output_sequence_length=self.NUM_UNIQUE_ITEMS)
        data = o.tvec(data)
        return data
    
    def sere_request(self, preprocessed_data):
        payload = {
            "instances": preprocessed_data.numpy().tolist()
            }
        
            
        # Implement the iterative thresholding logic
        STATE = True
        threshold = 0.5
        min_threshold = 0.0
        tolerance = 1e-6
        max_iterations = 10  # Limit the number of iterations
        iterations = 0

        while STATE and iterations < max_iterations:
            response = requests.post(self.url, json=payload)
            # Check for errors
            if response.status_code == 200:
                # Process the response (predictions)
                predictions = response.json()["predictions"]
            
                predictions = tensorflow.where(numpy.array(predictions) >= threshold, 1, 0)
                predicted_labels = []

                for pred in predictions:
                    indices = numpy.where(pred == 1)[0]
                    predicted_ngrams = [self.labels_vocab[i] for i in indices]
                    predicted_labels.extend(predicted_ngrams)

                predicted_labels = [i for i in predicted_labels if i != '<PAD>']

                if predicted_labels:
                    STATE = False
                else:
                    threshold -= 0.1
                    iterations += 1
                    if threshold < min_threshold + tolerance:
                        print("Threshold too low, stopping iterations.")
                        break
                    
                # print("Predictions:", predicted_labels)
            else:
                print("Error:", response.text)
        # Final output
        if predicted_labels:
            print(predicted_labels)
        else:
            print("No valid predictions found after threshold adjustments.")

    
    @staticmethod
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
        # try:
            # res = parser.parse_args()
        # except argparse.ArgumentError:
            # print('Got an arguemntError')
        # return res 
        return parser.parse_args()
        
        
    
if __name__ == '__main__':
    query = Query()
    args = query.parse_arguments()
    assert args.predict, 'no string passed for prediction'
    data = query.prepare_data(args.predict)
    query.sere_request(data)