import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tests.import_from_source as test_import


import pytorch3dunet
# import pytorch3dunet.unet3d as u3  # !
# from pytorch3dunet import predict
# from pytorch3dunet import train

# from pytorch3dunet.unet3d import trainer as trainer
# from pytorch3dunet.unet3d.config import _load_config_yaml

from torchsummary import summary
import torch
from torch.utils.data import Dataset, DataLoader


# these lines are required to test dataset / dataloader code from datasets.utils
from pytorch3dunet.unet3d.utils import get_logger, get_class
from pytorch3dunet.datasets.utils import _loader_classes, get_slice_builder
logger = get_logger('Dataset') 



# original logic that loads h5 files and makes them into pytorch Dataset objects
# from l. 125 of datasets.hdf5

def create_datasets(cls, dataset_config, phase):
    phase_config = dataset_config[phase]

    # load data augmentation configuration
    transformer_config = phase_config['transformer']
    # load slice builder config
    slice_builder_config = phase_config['slice_builder']
    # load files to process
    file_paths = phase_config['file_paths']
    # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
    # are going to be included in the final file_paths
    file_paths = cls.traverse_h5_paths(file_paths)

    datasets = []
    for file_path in file_paths:
        try:
            logger.info(f'Loading {phase} set from: {file_path}...')
            dataset = cls(file_path=file_path,
                          phase=phase,
                          slice_builder_config=slice_builder_config,
                          transformer_config=transformer_config,
                          raw_internal_path=dataset_config.get('raw_internal_path', 'raw'),
                          label_internal_path=dataset_config.get('label_internal_path', 'label'),
                          weight_internal_path=dataset_config.get('weight_internal_path', None),
                          global_normalization=dataset_config.get('global_normalization', None))
            datasets.append(dataset)
        except Exception:
            logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
    return datasets

# hdf5 -> torch.Tensor logic
# original definition on l. 227 of datasets.utils

def get_test_dataset(config):
    """
    Returns test Dataset.

    :return: Dataset objects
    """

    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating test set loaders...')

    # get dataset class
    dataset_cls_str = loaders_config.get('dataset', None)
    if dataset_cls_str is None:
        dataset_cls_str = 'StandardHDF5Dataset'
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")
    dataset_class = _loader_classes(dataset_cls_str)

    test_datasets = dataset_class.create_datasets(loaders_config, phase='test')
    return test_datasets
    
    
def get_test_loaders(test_datasets, config):
    """ Given a test_dataset, wrap it in a DataLoader and return
        the dataloaders.
    """
    loaders_config = config['loaders']
    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for the dataloader: {num_workers}')

    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for dataloader: {batch_size}')

    # use generator in order to create data loaders lazily one by one
    for test_dataset in test_datasets:
        logger.info(f'Loading test set from: {test_dataset.file_path}...')
        if hasattr(test_dataset, 'prediction_collate'):
            collate_fn = test_dataset.prediction_collate
        else:
            collate_fn = default_prediction_collate

        yield DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         collate_fn=collate_fn)






def main():
    sam = False  # (False if ben) thinking we could maintain different filepaths here -- or we could copy the whole notebook and maintain two different working notebooks.

    if sam:
        data_dir ='/scratch/groups/jyeatman/samjohns-projects/unet3d/' 
        config_dir = f'{data_dir}/pytorch-3dunet/resources/3DUnet_confocal_boundary'
        test_config_filename = 'test_config.yml'
        train_config_filename = 'train_config.yml'
        test_example_fp = f'{data_dir}/data/osfstorage-archive-test/N_511_final_crop_ds2.h5'
        train_config_fp = f'{data_dir}/pytorch-3dunet/resources/3DUnet_confocal_boundary/train_config.yml'

    else: 
        data_dir = '/home/weiner/bparker/code/models/pytorch-3dunet'
        h5_dir = '/home/weiner/HCP/projects/CNL_scalpel/h5'
        config_dir = './3DUnet_confocal_boundary'
        test_config_filename = 'mris_config_test.yml'
        train_config_filename = 'mris_config_train.yml'
        test__fp = f'{data_dir}/data/osfstorage-archive-test/N_511_final_crop_ds2.h5'
        train_fp = f'{data_dir}/mris_config_train.yml'


    # load config file
    train_config = pytorch3dunet.unet3d.config._load_config_yaml(train_fp)

    # modify trainer transforms
    train_config_modified = train_config.copy()
    train_config_modified['transformer'] =  {'raw': [{'name': 'Standardize'},
     {'name': 'ToTensor', 'expand_dims': True}],
    'label': [{'name': 'RandomFlip'},
     {'name': 'RandomRotate90'},
     {'name': 'RandomRotate',
      'axes': [[2, 1]],
      'angle_spectrum': 45,
      'mode': 'reflect'},
     {'name': 'StandardLabelToBoundary', 'append_label': True},
     {'name': 'ToTensor', 'expand_dims': False}]}
    
    # create trainer
    trainer = pytorch3dunet.unet3d.trainer.create_trainer(train_config_modified)

    # get model
    model = pytorch3dunet.unet3d.u3.model.get_model(train_config['model'])

    # load subjects
    train_hcp_subject_h5s = sorted(os.listdir(f"{h5_dir}/train"))






if __name__ == '__main__':
    main()
