import os
import argparse
import numpy as np
from torch.backends import cudnn
from loader import get_loader
from solver_RED_distill import Solver
from config import get_config
from utils import save_config, prepare_dirs


def main(config):
    cudnn.benchmark = True

    # ensure directories are setup
    prepare_dirs(config)


    data_loader = get_loader(mode=config.mode,
                             load_mode=config.load_mode,
                             saved_path=config.saved_path,
                             saved_path1=config.saved_path1,
                             test_patient=config.test_patient,
                             patch_n=(config.patch_n if config.mode=='train' else None),
                             patch_size=(config.patch_size if config.mode=='train' else None),
                             transform=config.transform,
                             batch_size=(config.batch_size if config.mode=='train' else 1),
                             num_workers=config.num_workers,
                             shuffle=config.shuffle)

    solver = Solver(config, data_loader)
    if config.mode == 'train':
        save_config(config)
        solver.train()

    elif config.mode == 'test':
        solver.test()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)

