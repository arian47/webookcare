import numpy
import tensorflow
import logging
import pathlib
import os
from webookcare.tools import save_models
from tensorflow.keras.backend import clear_session
import gc

current_dir = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = os.path.join(
    current_dir, 
    "saved_models/location_ranking").replace('\\', '/')
# app = typer.Typer()

numpy.set_printoptions(threshold=numpy.inf)
# Set TensorFlow logging level to ERROR
tensorflow.get_logger().setLevel('ERROR')
# Suppress warnings from the Python logging module
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# @app.command()
def check_shape(input_):
    """
    Ensures that the input tensor has at least two dimensions. If the input tensor
    has only one dimension, it is expanded to two dimensions by adding an extra dimension
    at the start.

    Parameters
    ----------
    input_ : tensorflow.Tensor
        The input tensor to check and modify.

    Returns
    -------
    tensorflow.Tensor
        The input tensor with at least two dimensions.
    """
    if input_.ndim == 1:
        input_ = tensorflow.expand_dims(input_, 0)
    elif input_.ndim >= 1:
        pass
    return input_

# @app.command()
def load_model(name):
    """
    Loads a pre-trained model from disk. If loading fails due to resource exhaustion or internal error,
    the session is cleared, garbage collection is performed, and the model is attempted to load again.

    Returns
    -------
    tensorflow.keras.Model
        The loaded Keras model.
    """
    try:
        model = save_models.load(DEFAULT_MODEL_PATH)
    except tensorflow.errors.ResourceExhaustedError or \
        tensorflow.errors.InternalError:
            tensorflow.keras.backend.clear_session()
            gc.collect()
            model = save_models.load(DEFAULT_MODEL_PATH)
    return model

# @app.command()
def predict(data, model_name='location_ranking'):
    # data = tensorflow.cast(data, 
                        #    dtype=tensorflow.float32)
    data = tuple(map(float, data.strip('()').split(', ')))
    data = tensorflow.constant(data)
    data = check_shape(data)
    model = load_model(model_name)
    
    # predicting labels
    if "serving_default" in model.signatures:
        infer = model.signatures['serving_default']
        predictions = infer(data)
        output_key = list(predictions.keys())[0]
        predictions = predictions[output_key]
        
    else:
        predictions = model.predict(data)
    
    predictions = predictions[0]