import os
import math
from os import path as osp
from loguru import logger
from torch.utils.data.dataset import Dataset

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import DataLoader

from src.datasets.feature_dataset import FeatureDataset, FeatureDataset_sp
from src.utils.triplet_sampler import TripletSampler

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

        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
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
            if self.data_source == 'delg_gldv2':
                
                self.train_dataset = FeatureDataset(data_dir=self.config.DATASET.DELG_GLDV2.TRAIN_DATA_ROOT,
                                                          sample_file=self.config.DATASET.DELG_GLDV2.TRAIN_TXT,
                                                          max_sequence_len=self.config.DATASET.MAX_SEQ_LEN
                                                        )

                # self.val_dataset = FeatureDataset_train(data_dir=self.config.DATASET.DELG_GLDV2.TRAIN_DATA_ROOT,
                #                                         sample_file=self.config.DELG_GLDV2.TRAIN_TXT,
                #                                         max_sequence_len=self.config.MAX_SEQ_LEN
                #                                         )
            elif self.data_source == 'sp_gldv2':
                self.train_dataset = FeatureDataset_sp(data_dir=self.config.DATASET.SP_GLDV2.TRAIN_DATA_ROOT,
                                                        sample_file=self.config.DATASET.SP_GLDV2.TRAIN_TXT,
                                                        max_sequence_len=self.config.DATASET.MAX_SEQ_LEN,
                                                        gnd_data=None,
                                                        is_train=True
                                                        )
            else:
                raise NotImplementedError('data source must be in [\'Pittsburgh\', \'GoogleLandMarkV2\', \'GoogleLandMarkV2_triplet\']')
            logger.info('Train Dataset loaded!')

        elif stage == 'validate':
            if self.data_source == 'delg_gldv2':
                ...
            else:
                raise NotImplementedError()
            logger.info('Val Dataset loaded!')
        elif stage == 'test':
            logger.info('Test Dataset loaded!')

    def train_dataloader(self):
        """ Build training dataloader. """
        logger.info(' train dataloader')
        if self.data_source == 'delg_gldv2':
            # 定义采样器
            train_sampler = TripletSampler(labels=self.train_dataset.targets, 
                        batch_size=self.train_loader_params['batch_size'], 
                        nn_inds_path=self.config.DATASET.DELG_GLDV2.TRAIN_NN_INDS)
        
            train_loader = DataLoader(self.train_dataset, 
                                    batch_sampler=train_sampler, 
                                    num_workers=self.train_loader_params['num_workers'], 
                                    pin_memory=self.train_loader_params['pin_memory']
                                    )
        elif self.data_source == 'sp_gldv2':
            train_sampler = TripletSampler(labels=self.train_dataset.targets, 
                        batch_size=self.train_loader_params['batch_size'], 
                        nn_inds_path=self.config.DATASET.SP_GLDV2.TRAIN_NN_INDS)
        
            train_loader = DataLoader(self.train_dataset, 
                                    batch_sampler=train_sampler, 
                                    num_workers=self.train_loader_params['num_workers'], 
                                    pin_memory=self.train_loader_params['pin_memory']
                                    )
        else:
            ...
        return train_loader
    
    def val_dataloader(self):
        """ Build validation dataloader. """