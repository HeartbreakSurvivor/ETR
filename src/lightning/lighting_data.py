import os
import math
from collections import abc
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from os import path as osp
from pathlib import Path
from joblib import Parallel, delayed
from loguru import logger

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    RandomSampler,
    dataloader
)

from src.datasets.pittsburgh import pitts_collate_fn, PittsburghTripletDataset, WholePittsburghDataset
from src.datasets.googleLandmarkv2 import gld_collate_fn, default_scale_transform,\
     GLDv2ClassificationDataset, GLDv2TripletDataset, GLDv2ValidationDataset

class MultiSceneDataModule(pl.LightningDataModule):
    """
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        # Train and Val should from the same data source
        self.config = config
        self.data_source = config.DATASET.DATA_SOURCE

        if self.data_source == 'Pittsburgh':
            self.collate_fn = pitts_collate_fn
        elif self.data_source == 'GoogleLandMarkV2_triplet':
            self.collate_fn = gld_collate_fn

        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            # 'pin_memory': getattr(args, 'pin_memory', True)
            'pin_memory': True # to speed up
        }
        self.val_loader_params = {
            'batch_size': 16, # must be 1 to build feature index
            'shuffle': False,
            'num_workers': args.num_workers,
            # 'pin_memory': getattr(args, 'pin_memory', True)
            'pin_memory': True # to speed up
        }
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': True
        }

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, 'validate' in validating phase, and 'test' in testing phase.
        """
        logger.info('====> lightning data module stage: {}', stage)
        assert stage in ['fit', 'validate', 'test'], "stage must be either fit or test"
        if stage == 'fit':
            if self.data_source == 'Pittsburgh':
                
                self.train_dataset = PittsburghTripletDataset(
                    root_dir=self.config.DATASET.PITTS.TRAIN_DATA_ROOT,
                    mode=self.config.DATASET.PITTS.MODE,
                    scale=self.config.DATASET.PITTS.SCALE,
                    nNeg=self.config.DATASET.PITTS.NEGATIVES,
                    margin=self.config.DATASET.PITTS.MARGIN,
                    nNegSample=self.config.DATASET.PITTS.NEGATIVE_SAMPLES,
                    cache_file=self.config.DATASET.PITTS.CACHE_FILE,
                )
                self.val_dataset = WholePittsburghDataset(
                    root_dir=self.config.DATASET.PITTS.TRAIN_DATA_ROOT,
                    mode='val',
                    scale=self.config.DATASET.PITTS.SCALE,
                    onlyDB=False,
                )
            elif self.data_source == 'GoogleLandMarkV2': # for classification task
                logger.info('load GoogleLandMarkV2 train dataset')
                
                self.train_dataset = GLDv2ClassificationDataset(
                    root_dir=self.config.DATASET.GLDV2.TRAIN_DATA_ROOT,
                    mode='train',
                    transform=default_scale_transform # just for train
                    )
                logger.info('GoogleLandMarkV2 train dataset len: {}', len(self.train_dataset))
                
                logger.info('load GoogleLandMarkV2 val dataset')
                self.val_dataset = GLDv2ClassificationDataset(
                    root_dir=self.config.DATASET.GLDV2.TRAIN_DATA_ROOT,
                    mode='val')
                logger.info('GoogleLandMarkV2 val dataset len: {}', len(self.val_dataset))

            elif self.data_source == 'GoogleLandMarkV2_triplet': # for metric learning task
                self.train_dataset = GLDv2TripletDataset(
                    root_dir=self.config.DATASET.GLDV2.TRAIN_DATA_ROOT,
                    num_query=self.config.DATASET.GLDV2.QUERY,
                    num_positive=self.config.DATASET.GLDV2.POSITIVES,
                    num_negative=self.config.DATASET.GLDV2.NEGATIVES
                    )
                self.val_dataset = GLDv2ValidationDataset(
                    root_dir=self.config.DATASET.GLDV2.TRAIN_DATA_ROOT)
            else:
                raise NotImplementedError('data source must be in [\'Pittsburgh\', \'GoogleLandMarkV2\', \'GoogleLandMarkV2_triplet\']')
            logger.info('Train Dataset loaded!')

        elif stage == 'validate':
            if self.data_source == 'Pittsburgh':
                ...
            elif self.data_source == 'GoogleLandMarkV2': # for classification task
                ...
            elif self.data_source == 'GoogleLandMarkV2_triplet': # for metric learning task
                ...
            else:
                raise NotImplementedError('data source must be in [\'Pittsburgh\', \'GoogleLandMarkV2\', \'GoogleLandMarkV2_triplet\']')
            logger.info('Val Dataset loaded!')

        elif stage == 'test':
            if self.data_source == 'Pittsburgh':
                self.test_dataset = WholePittsburghDataset(
                    root_dir=self.config.DATASET.PITTS.TRAIN_DATA_ROOT,
                    mode='test',
                    scale=self.config.DATASET.PITTS.SCALE,
                    onlyDB=False)
            elif self.data_source == 'GoogleLandMarkV2':
                pass
            elif self.data_source == 'GoogleLandMarkV2_triplet': # for metric learning task
                pass
            else:
                raise NotImplementedError('data source must be in [\'Pittsburgh\', \'GoogleLandMarkV2\', \'GoogleLandMarkV2_triplet\']')
            logger.info('Test Dataset loaded!')

    def train_dataloader(self):
        """ Build training dataloader. """
        logger.info(' train dataloader')
        if self.data_source == 'GoogleLandMarkV2':
            # set shuffle to True
            # 定义一个分布式采样器试试
            logger.info(' DistributedSampler for GLDv2')
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
            dataloader = DataLoader(self.train_dataset, sampler=sampler, shuffle=False, **self.train_loader_params)
        else:
            dataloader = DataLoader(self.train_dataset, collate_fn=self.collate_fn, **self.train_loader_params)
        return dataloader
    
    def val_dataloader(self):
        """ Build validation dataloader. """
        logger.info('val dataloader')
        dataloader = DataLoader(self.val_dataset, **self.val_loader_params)
        return dataloader

    def test_dataloader(self, *args, **kwargs):
        """ Build test dataloader. """
        return DataLoader(self.test_dataset, **self.test_loader_params)

    def predict_dataloader(self):
        """ Build predict dataloader. """
        ...