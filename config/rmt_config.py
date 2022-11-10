import os
from yacs.config import CfgNode as CN

_CN = CN()

############## RMT Configuration ##############
_CN.RMT = CN()

_CN.RMT.HEADS = 8
_CN.RMT.LAYER_NAMES = ['self', 'cross'] * 3 # depth
_CN.RMT.GLOBAL_DIM = 2048
_CN.RMT.EMBED_DIM = 128
_CN.RMT.FFN_DIM = 1024
_CN.RMT.SEQ_LEN = 502 # 序列长度
_CN.RMT.MHA_DROPOUT = 0.0
_CN.RMT.FFN_DROPOUT = 0.0
_CN.RMT.POOL = 'avg'  # options: ['cls', 'avg', 'max', 'gem']
_CN.RMT.NORMALIZE_BEFORE = False
_CN.RMT.ACTIVATION = 'relu'  # options: ['relu', 'gelu', 'glu', 'elu']
_CN.RMT.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.RMT.POSITION_EMBEDDING_TYPE = 'default'

############## Dataset Configuration ############
_CN.DATASET = CN()
# 1. training 
_CN.DATASET.DATA_SOURCE = 'delg_gldv2' # options: ['Pittsburgh', 'GoogleLandMarkV2', '']
_CN.DATASET.MAX_SEQ_LEN = 500

_CN.DATASET.DELG_GLDV2 = CN()  # (optional directory for poses)
_CN.DATASET.DELG_GLDV2.TRAIN_TXT = '/root/halo/RerankingTransformer/RRT_GLD/data/gldv2/train.txt'
_CN.DATASET.DELG_GLDV2.TRAIN_DATA_ROOT = '/root/halo/RerankingTransformer/RRT_GLD/data/gldv2/delg_r50_gldv2'
_CN.DATASET.DELG_GLDV2.TRAIN_NN_INDS = '/root/halo/RerankingTransformer/RRT_GLD/data/gldv2/nn_inds_r50_gldv2.pkl'
_CN.DATASET.DELG_GLDV2.SAMPLER = 'triplet'

# 2. for validation

################# Pitts30k ####################
_CN.DATASET.VAL_QUERY_ROOT = '/datasets/Pitts30k/test/delg_feats/query'
_CN.DATASET.VAL_QUERY_TXT = '/datasets/Pitts30k/test/pitts30k_query_c.txt'

_CN.DATASET.VAL_DB_ROOT = '/datasets/Pitts30k/test/delg_feats/index'
_CN.DATASET.VAL_DB_TXT = '/datasets/Pitts30k/test/pitts30k_db_c.txt'

_CN.DATASET.VAL_GND_FILE = '/datasets/Pitts30k/test/pitts30k_gt.npy'
_CN.DATASET.VAL_NN_INDS = '/datasets/Pitts30k/test/delg_feats/delg_pitts30k_test_rank_index.npy'
_CN.DATASET.VAL_BATCH_SIZE = 192
_CN.DATASET.VAL_WORKERS = 16

############## Trainer Configuration ############
_CN.TRAINER = CN()

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
_CN.TRAINER.SEED = 42 # need change when resume from a checkpoint

_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

_CN.TRAINER.EPOCHS = 30
_CN.TRAINER.MAX_NORM = 0.0 # [0.1]
_CN.TRAINER.CUDNN_FLAG = 'benchmark' # [0.1]

# loss
_CN.TRAINER.LOSS = 'BCE'  # ['BCE', 'Triplet', 'CrossEntropy', 'ArcFace', 'ListWise']

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [SGD, adam, adamw]
_CN.TRAINER.TRUE_LR = 0.001 # this will be calculated automatically at runtime

_CN.TRAINER.ADAM_DECAY = 0.005  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.001

_CN.TRAINER.SGD_DECAY = 0.2 # SGD: for sgd
_CN.TRAINER.SGD_MOMENTUM = 0.9

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.01 # 0.01
_CN.TRAINER.WARMUP_STEP = 4000

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'CosineAnnealing'  # [LambdaLR, MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.SCHEDULER_FREQUENCY = 1 # 1 corresponds to updating the learning rate after every epoch/step.

# for MultiStepLR
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5

# for CosineAnnealing
_CN.TRAINER.COSA_TMAX = _CN.TRAINER.EPOCHS  # COSA: CosineAnnealing

# for ExponentialLR
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
