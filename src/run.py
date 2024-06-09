import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname('__file__'), '.'))
src_path = os.path.join(project_root, 'tf_models')

if src_path not in sys.path:
    sys.path.append(src_path)

from tf_models.model_builder import *

def run(args):
    model, full_opt, opt_dict = build_model(args)
    return model, full_opt, opt_dict