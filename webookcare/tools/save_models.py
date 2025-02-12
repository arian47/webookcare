from tensorflow.keras.backend import clear_session
import tensorflow
import pathlib
import gc
import pickle

def create_dirs(spec=None):
    if not spec:
        dirs = dict(
            base_path=pathlib.Path('saved_models'),
        )
        for i in dirs:
            if not dirs[i].exists():
                dirs[i].mkdir(parents=True)
        return dirs
    else:
        spec.mkdir(parents=True)
        return None

def save(model, name, save_type):
    base_path = create_dirs().get('base_path').joinpath(name)
    if save_type == 'weigths':
        model.save_weights(
            base_path.joinpath('MLP_int_1g_bigram_multihot_labels.tf'))
    elif save_type == 'checkpoints':
        checkpoints_dir = base_path.joinpath('checkpoints')
        create_dirs(checkpoints_dir)
        checkpoint = tensorflow.train.Checkpoint(model=model)
        checkpoint.save(file_prefix=checkpoints_dir)
    elif save_type == 'default':
        tensorflow.saved_model.save(model, base_path)
        
def load(model_path):
    """
    Loads a pre-trained model from disk. If loading fails due to resource exhaustion or internal error,
    the session is cleared, garbage collection is performed, and the model is attempted to load again.

    Returns
    -------
    tensorflow.keras.Model
        The loaded Keras model.
    """
    try:
        # base_path = create_dirs().get('base_path').resolve()

        # Load the saved model (if using SavedModel format)
        # model_path = base_path.joinpath(f'{name}')
        loaded_model = tensorflow.saved_model.load(model_path)
    except tensorflow.errors.ResourceExhaustedError or \
        tensorflow.errors.InternalError:
            tensorflow.keras.backend.clear_session()
            gc.collect()
            model = loaded_model.load(model_path)
    return loaded_model

def save_ml(model, model_path, type:str='pickle'):
    if type == 'pickle':
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
    # elif type == 'surprise':
        # surprise.dump.dump(model_path, model)


def load_ml(model_path, type:str='pickle'):
    if type == 'pickle':
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    # elif type == 'surprise':
        # model = surprise.dump.load(model_path)
    return model
    