import os
from pickle import TRUE
from yacs.config import CfgNode as CN

_CN = CN()

############## MSFFT Configuration ##############
_CN.MSFFT = CN()
_CN.MSFFT.BACKBONE_TYPE = 'ResNetFPN'
_CN.MSFFT.RESNET_TYPE = 18 # options: [18, 34, 50, 101]

_CN.MSFFT.ATTN_MAP_LEVEL = 'mutual_patch_level' # options: ['pixel_level', 'single_patch_level', 'mutual_patch_level']

_CN.MSFFT.GLOBAL_FEAT_DIM = 2048 # options: [64, 128, 256, 512, 1024, 2048, 4096]
_CN.MSFFT.TRANSFORMER_FEAT_DIM = 256 

# 1. MSFFT backbone (local feature CNN) config
_CN.MSFFT.RESNETFPN = CN()
_CN.MSFFT.RESNETFPN.PRETRAINED = True
_CN.MSFFT.RESNETFPN.PROGRESS = True # whether or not to display a progress bar to stderr

# 2. MSFFT coarse transformer configuration
_CN.MSFFT.TRANSFORMER_COARSE = CN()
_CN.MSFFT.TRANSFORMER_COARSE.IMAGE_SIZE = (15, 20) # options [(15, 20), (30, 40), (60, 80)]
_CN.MSFFT.TRANSFORMER_COARSE.CHANNELS = 2048 # reset50 layer4's output channel
_CN.MSFFT.TRANSFORMER_COARSE.PATCH_SIZE = (5, 5)
_CN.MSFFT.TRANSFORMER_COARSE.HIDDEN_DIM = 512 # -1 means don't change hidden dimension 
_CN.MSFFT.TRANSFORMER_COARSE.FFN_DIM = 256 # reduce dimension before go through transformer block
_CN.MSFFT.TRANSFORMER_COARSE.DEPTH = 4
_CN.MSFFT.TRANSFORMER_COARSE.HEADS = 8
_CN.MSFFT.TRANSFORMER_COARSE.DROPOUT = 0.1
_CN.MSFFT.TRANSFORMER_COARSE.EMB_DROPOUT = 0.1
_CN.MSFFT.TRANSFORMER_COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.MSFFT.TRANSFORMER_COARSE.POSITION_EMBEDDING_TYPE = 'default'

# 3. MSFFT fine transformer config
_CN.MSFFT.TRANSFORMER_FINE0 = CN()
_CN.MSFFT.TRANSFORMER_FINE0.IMAGE_SIZE = (30, 40) # options [(15, 20), (30, 40), (60, 80)]
_CN.MSFFT.TRANSFORMER_FINE0.CHANNELS = 256
_CN.MSFFT.TRANSFORMER_FINE0.PATCH_SIZE = (5, 5)
_CN.MSFFT.TRANSFORMER_FINE0.HIDDEN_DIM = 256 # -1 means don't change hidden dimension 
_CN.MSFFT.TRANSFORMER_FINE0.FFN_DIM = 128 # reduce dimension before go through transformer block
_CN.MSFFT.TRANSFORMER_FINE0.DEPTH = 4
_CN.MSFFT.TRANSFORMER_FINE0.HEADS = 8
_CN.MSFFT.TRANSFORMER_FINE0.DROPOUT = 0.1
_CN.MSFFT.TRANSFORMER_FINE0.EMB_DROPOUT = 0.1
_CN.MSFFT.TRANSFORMER_FINE0.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.MSFFT.TRANSFORMER_FINE0.POSITION_EMBEDDING_TYPE = 'default'

_CN.MSFFT.TRANSFORMER_FINE1 = CN()
_CN.MSFFT.TRANSFORMER_FINE1.IMAGE_SIZE = (60, 80) # options [(15, 20), (30, 40), (60, 80)]
_CN.MSFFT.TRANSFORMER_FINE1.CHANNELS = 256
_CN.MSFFT.TRANSFORMER_FINE1.PATCH_SIZE = (5, 5)
_CN.MSFFT.TRANSFORMER_FINE1.HIDDEN_DIM = 128 # -1 means don't change hidden dimension
_CN.MSFFT.TRANSFORMER_FINE1.DEPTH = 4
_CN.MSFFT.TRANSFORMER_FINE1.HEADS = 8
_CN.MSFFT.TRANSFORMER_FINE1.FFN_DIM = 64 # reduce dimension before go through transformer block
_CN.MSFFT.TRANSFORMER_FINE1.DROPOUT = 0.1
_CN.MSFFT.TRANSFORMER_FINE1.EMB_DROPOUT = 0.1
_CN.MSFFT.TRANSFORMER_FINE1.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.MSFFT.TRANSFORMER_FINE1.POSITION_EMBEDDING_TYPE = 'default'

############## Dataset Configuration ############
_CN.DATASET = CN()
# data config

# 1. training and validation
_CN.DATASET.DATA_SOURCE = 'Pittsburgh' # options: ['Pittsburgh', 'GoogleLandMarkV2', 'GoogleLandMarkV2_triplet']

_CN.DATASET.PITTS = CN()  # (optional directory for poses)
_CN.DATASET.PITTS.TRAIN_DATA_ROOT = '/dataset/GoogleLandmarkV2/train_clean'
_CN.DATASET.PITTS.MODE = 'train' # options: ['train', 'val', 'test']
_CN.DATASET.PITTS.SCALE = '30k' # options: ['30k', '250k']
_CN.DATASET.PITTS.MARGIN = 0.1
_CN.DATASET.PITTS.NEGATIVES = 10 # number negative for training
_CN.DATASET.PITTS.NEGATIVE_SAMPLES = 1000 # number of negatives to randomly sample
_CN.DATASET.PITTS.CACHE_FILE = os.getcwd() + '/output/cache/{}_feat_cache.hdf5'.format(_CN.DATASET.PITTS.MODE)
_CN.DATASET.PITTS.TRANSFORM = None

# 2. test

# 3. dataset config
# general options
_CN.DATASET.TRANSFORM = None

############## Trainer Configuration ############
_CN.TRAINER = CN()
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# loss
_CN.TRAINER.LOSS = 'Triplet'  # ['Triplet', 'CrossEntropy', 'ArcFace', 'ListWise']

# for TripletMarginLoss
_CN.TRAINER.TRIPLET_MARGIN = 0.1 
_CN.TRAINER.TRIPLET_P = 2 # ∈[0,∞] The norm degree for pairwise distance.
_CN.TRAINER.TRIPLET_REDUCTION = 'mean' # ['none', 'mean', 'sum'] Specifies the reduction to apply to the output

# for ArcFace

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [SGD, adam, adamw]
_CN.TRAINER.TRUE_LR = 0.005  # this will be calculated automatically at runtime

_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.05

_CN.TRAINER.SGD_DECAY = 0 # SGD: for sgd
_CN.TRAINER.SGD_MOMENTUM = 0.9

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.1
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'CosineAnnealing'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.SCHEDULER_FREQUENCY = 1    # 1 corresponds to updating the learning rate after every epoch/step.

# for MultiStepLR
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
# for CosineAnnealing
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
# for ExponentialLR
_CN.TRAINER.ELR_GAMMA = 0.88  # ELR: ExponentialLR, this value for 'step' interval

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, DEGENSAC, MAGSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 42

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
