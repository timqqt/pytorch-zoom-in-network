import os 
import numpy as np

def load_annotations(data_dir):
    '''
    load the path of annotations
    '''
    at_dirs = np.load(os.path.join(data_dir, 'annotation_names.npy'))
    train_at, valid_at, test_at = at_dirs[0], at_dirs[1], at_dirs[2]
    return train_at, valid_at, test_at