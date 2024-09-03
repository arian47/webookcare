import tensorflow
import pathlib


def create_dirs(spec=None):
    # checkpoints_dir=pathlib.Path('saved_models/checkpoints/'),
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
        model.save_weights(base_path.joinpath('MLP_int_1g_bigram_multihot_labels.tf'))
    elif save_type == 'checkpoints':
        checkpoints_dir = base_path.joinpath('checkpoints')
        create_dirs(checkpoints_dir)
        checkpoint = tensorflow.train.Checkpoint(model=model)
        checkpoint.save(file_prefix=checkpoints_dir)
    elif save_type == 'default':
        tensorflow.saved_model.save(model, base_path)
        
def load(name):
    base_path = create_dirs().get('base_path')
    # checkpoint_dir = './checkpoints/MLP_int_1g_bigram_multihot_labels'
    # checkpoint = tensorflow.train.Checkpoint(model=model)
    # checkpoint.restore(tensorflow.train.latest_checkpoint(checkpoint_dir))

    # Load the saved model (optional if using SavedModel format)
    # loaded_model = tensorflow.saved_model.load(model_save_path)
    model_path = list(base_path.rglob(f'{name}'))
    assert len(model_path) == 1, 'too many models found!'
    loaded_model = tensorflow.saved_model.load(model_path[0])
    
    # Example usage after loading
    # infer = loaded_model.signatures["serving_default"]
    return loaded_model

        
        