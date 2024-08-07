import os
import json
import torch
from config import get_config

def save_config(config):
    model_name = config.save_path
    filename = '_params.json'
    param_path = os.path.join(config.save_path, filename)
    print("[*] Model Checkpoint Dir: {}".format(config.save_path))
    print("[*] Param Path: {}".format(param_path))

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def prepare_dirs(config):
    if not os.path.exists(config.save_path):
        ckpt_path = os.path.join(config.save_path,'ckpt')
        fig_path = os.path.join(config.save_path, 'fig')
        os.makedirs(config.save_path)
        os.makedirs(ckpt_path)
        os.makedirs(fig_path)
        print('Create path : {}'.format(config.save_path))


def save_fig(config):
    if config.result_fig:
        fig_path = os.path.join(config.save_path, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))
