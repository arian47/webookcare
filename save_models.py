import tensorflow
import pathlib

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
        
def load(name):
    base_path = create_dirs().get('base_path').absolute()

    # Load the saved model (if using SavedModel format)
    model_path = base_path.joinpath(f'{name}')
    loaded_model = tensorflow.saved_model.load(model_path)
    
    return loaded_model

        
        