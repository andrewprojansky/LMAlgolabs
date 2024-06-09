import os
import sys


def prepare_file_system():
    # Ensure the 'src' directory is in the Python path
    project_root = os.path.abspath(os.path.join(os.path.dirname('__file__'), '.'))
    src_path = os.path.join(project_root, 'src')

    data_path = os.path.join(src_path, 'data')
    os.makedirs(data_path, exist_ok=True)

    if src_path not in sys.path:
        sys.path.append(src_path)

prepare_file_system()


from src import VMC
from run import run
from my_types import SaveType, InteractionType, InvalidInteractionError, Result, Ok, Err
from helpers import TransformerConfigDescription
from helpers import save_rnn, save_ptf, save_to_file
from helpers import TransformerConfig, RNNConfig, VMCConfig
from helpers import get_widget_group, ModelType, RydbergConfig, TrainConfig
from helpers import RydbergConfigDescription, TrainConfigDescription, VMCConfigDescription


from rnn_model import VMC, get_model
from interactions import get_interactions