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
# import train

from tensorflow.keras.backend import clear_session
import gc

app = typer.Typer()

numpy.set_printoptions(threshold=numpy.inf)
# Set TensorFlow logging level to ERROR
tensorflow.get_logger().setLevel('ERROR')
# Suppress warnings from the Python logging module
logging.getLogger('tensorflow').setLevel(logging.ERROR)

@app.command()
def check_shape(input_):
    if input_.ndim == 1:
        input_ = tensorflow.expand_dims(input_, 0)
    elif input_.ndim >= 1:
        pass
    return input_

@app.command()
def load_model(name):
    try:
        model = webookcare.save_models.load(name)
    except tensorflow.errors.ResourceExhaustedError or \
        tensorflow.errors.InternalError:
            tensorflow.keras.backend.clear_session()
            gc.collect()
            model = webookcare.save_models.load(name)
    return model

@app.command()
def predict(data, model_name='location_ranking'):
    # data = tensorflow.cast(data, 
                        #    dtype=tensorflow.float32)
    data = tuple(map(float, data.strip('()').split(', ')))
    data = tensorflow.constant(data)
    data = check_shape(data)
    model = load_model(model_name)
    predict_fn = model.signatures["serving_default"]
    predictions = predict_fn(data)
    # predictions = tensorflow.cast(predictions, 
                                #   dtype=tensorflow.float32)
    print(predictions['output_1'])
    
if __name__ == '__main__':
    app()