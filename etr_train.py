import math
import argparse
from distutils.util import strtobool
from pickle import NONE
from loguru import logger

from torch.utils.data import DataLoader, RandomSampler, BatchSampler

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.plugins import DDPPlugin

from config.rmt_config import get_cfg_defaults
from src.lightning.lighting_data_rmt import MultiSceneDataModule
from lightning.lighting_etr import PL_ETR

from src.utils.evaluate import Evaluator
from src.datasets.feature_dataset import FeatureDataset

def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=192, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=16)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    # parser.add_argument(
    #     '--ckpt_path', type=str, default=None,
    #     help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    # parser.add_argument(
    #     '--disable_ckpt', action='store_true', 
    #     help='disable checkpoint saving (useful for debugging).')
    # parser.add_argument(
    #     '--parallel_load_data', action='store_true',
    #     help='load datasets in with multiple processes.')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()

    config = get_cfg_defaults()
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED for reproducibility
    pl.seed_everything(config.TRAINER.SEED)

    log_outdir = "output/logs/{}".format(args.exp_name)
    logger.add(log_outdir + "/file_{time}.log")

    # TensorBoard Logger
    TBlogger = TensorBoardLogger(save_dir='tb_logs', name=args.exp_name, default_hp_metric=False)
    # ckpt_dir = Path(logger.log_dir) / 'checkpoints'

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    logger.info(f"===> RMT DataModule initialized!")

    # val dataset
    query_set = FeatureDataset(data_dir=config.DATASET.VAL_QUERY_ROOT,
                                   sample_file=config.DATASET.VAL_QUERY_TXT,
                                   max_sequence_len=config.DATASET.MAX_SEQ_LEN,
                                   gnd_data=config.DATASET.VAL_GND_FILE
                                )
    index_set = FeatureDataset(data_dir=config.DATASET.VAL_DB_ROOT,
                                sample_file=config.DATASET.VAL_DB_TXT,
                                max_sequence_len=config.DATASET.MAX_SEQ_LEN
                            )                          
    query_loader = DataLoader(query_set, batch_size=config.DATASET.VAL_BATCH_SIZE, 
                                num_workers=args.num_workers, pin_memory=args.pin_memory)
    index_loader = DataLoader(index_set, batch_size=config.DATASET.VAL_BATCH_SIZE, 
                                num_workers=args.num_workers, pin_memory=args.pin_memory)

    evaluator = Evaluator(dataset_name='pitts30k', 
                    cache_nn_inds=config.DATASET.VAL_NN_INDS, # 这个代表的就是要被调整的NN文件
                    query_loader=query_loader,
                    index_loader=index_loader)

    ckpt = "/output/checkpoints/epoch13_acc_88_recall@1_84.21_recall@2_91.61_recall@10_93.78.ckpt"
    if ckpt:
        model = PL_ETR.load_from_checkpoint(ckpt,config=config, 
                                            evaluator=evaluator)
        logger.info(f"===> Load ETR from checkpoint..")
    else:
        model = PL_ETR(config=config, 
                    evaluator=evaluator, 
                    pretrained_ckpt=ckpt)

    logger.info(f"===> ETR LightningModule initialized!")

    # # Callbacks
    # # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(monitor='val/Recall@5', 
                                    mode='max',
                                    verbose=True,
                                    save_top_k=3,
                                    every_n_epochs=1,
                                    dirpath='/output/checkpoints/{}'.format(args.exp_name), # save checkpoint
                                    filename='{epoch}-{train_avg_loss:.2f}-{train_avg_acc:.2f}-{Recall@1:.4f}-{Recall@5:.4f}-{Recall@10:.4f}'
                                    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor, ckpt_callback]
    # if not args.disable_ckpt:
    #     callbacks.append(ckpt_callback)
    
    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=[2],
        # strategy='ddp',
        # accelerator="ddp",
        plugins=DDPPlugin(find_unused_parameters=True),
        # precision=16, 
        logger=TBlogger,
        max_epochs=config.TRAINER.EPOCHS,
        callbacks=callbacks,
        # val_check_interval=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=500,
        enable_checkpointing=True,
        enable_progress_bar=True,
        replace_sampler_ddp=False, # use customer sampler
        # auto_scale_batch_size='power',
        profiler='simple',
        # default_root_dir='/root/test/zhanghao/code/MSFFT/output/checkpoints' # save checkpoint
    )

    logger.info(f"===> Trainer initialized!")
    logger.info(f"===> Start training!")

    # for training 
    trainer.fit(model, datamodule=data_module)

    # for validating 
    # trainer.validate(model, dataloaders=data_module,
    #     ckpt_path='/work_dir/vpr/MSFFT/output/checkpoints/last.ckpt')

    # # for testing
    # trainer.test(model, dataloaders=data_module,
    #     ckpt_path='/work_dir/vpr/MSFFT/output/checkpoints/last.ckpt')

if __name__ == '__main__':
    main()