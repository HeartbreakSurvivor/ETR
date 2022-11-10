
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loguru import logger

# from config.rmt_config import get_cfg_defaults
from config.rmt_config_v1 import get_cfg_defaults

# need change if model has changed
from src.utils.evaluate import Evaluator
from src.model.rmt.rmt_matcher import ReMatcher
from src.datasets.general_dataset import PlaceDataset
from src.datasets.feature_dataset import FeatureDataset
from src.utils.misc import lower_config, load_checkpoint

TOP_N = [100]
N_RECALLS = [1, 5, 10, 15, 20, 25]
MSFFT_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

EVAL_DATASETS = {
    'pitts_30k': {
        'query_path': '/datasets/Pitts30k/test/pitts30k_query_c.txt',
        'query_dir': '/datasets/Pitts30k/test/delg_feats/query',
        'db_path': '/datasets/Pitts30k/test/pitts30k_db_c.txt',
        'db_dir': '/datasets/Pitts30k/test/delg_feats/index',
        'gt_path': '/datasets/Pitts30k/test/pitts30k_gt.npy',
        'rank_dir': '/datasets/Pitts30k/test/delg_feats/delg_pitts30k_test_rank_index.npy',
        'out_dir': '/output_results/pitts30k'
    },
    'tokyo247': {
        'query_path': '/datasets/Tokyo247/tokyo247_query_c.txt',
        'query_dir': '/datasets/Tokyo247/delg_feats/query',
        'db_path': '/datasets/Tokyo247/tokyo247_db_c.txt',
        'db_dir': '/datasets/Tokyo247/delg_feats/index',
        'gt_path': '/datasets/Tokyo247/tokyo247_gt.npy',
        'rank_dir': '/datasets/Tokyo247/delg_feats/delg_tokyo247_rank_index.npy',
        'out_dir': '/output_results/tokyo247'
    },
    'msls_cph': {
        'query_path': '/datasets/MSLS_val/cph/cph_query_c.txt',
        'query_dir': '/datasets/MSLS_val/cph/delg_feats/query',
        'db_path': '/datasets/MSLS_val/cph/cph_db_c.txt',
        'db_dir': '/datasets/MSLS_val/cph/delg_feats/index',
        'gt_path': '/datasets/MSLS_val/cph/cph_gt.npy',
        'rank_dir': '/datasets/MSLS_val/cph/delg_feats/cph_rank_index.npy',
        'out_dir': '/output_results/cph'
    },
    'msls_sf': {
        'query_path': '/datasets/MSLS_val/sf/sf_query_c.txt',
        'query_dir': '/datasets/MSLS_val/sf/delg_feats/query',
        'db_path': '/datasets/MSLS_val/sf/sf_db_c.txt',
        'db_dir': '/datasets/MSLS_val/sf/delg_feats/index',
        'gt_path': '/datasets/MSLS_val/sf/sf_gt.npy',
        'rank_dir': '/datasets/MSLS_val/sf/delg_feats/sf_rank_index.npy',
        'out_dir': '/output_results/sf'
    },
    'roxford5k': {
        'query_path': '/datasets/rOxford5k/roxford5k_query.txt',
        'query_dir': '/datasets/rOxford5k/delg_feats',
        'db_path': '/datasets/rOxford5k/roxford5k_index.txt',
        'db_dir': '/datasets/rOxford5k/delg_feats',
        'gt_path': '/datasets/rOxford5k/gnd_roxford5k.pkl',
        'rank_dir': '/datasets/rOxford5k/nn_inds_r50_gldv2.pkl',
        'out_dir': '/output_results/roxford5k'
    },
    'rparis6k': {
        'query_path': '/datasets/rPairs6k/rparis6k_query.txt',
        'query_dir': '/datasets/rPairs6k/delg_feats/query',
        'db_path': '/datasets/rPairs6k/rparis6k_index.txt',
        'db_dir': '/datasets/rPairs6k/delg_feats/index',
        'gt_path': '/datasets/rPairs6k/gnd_rparis6k.pkl',
        'rank_dir': '/datasets/rPairs6k/nn_inds_r50_gldv2.pkl',
        'out_dir': '/output_results/rPairs6k'
    },
}

def main():
    device = 3
    num_workers = 16
    batch_size = 192
    max_seq_len = 500
    pin_memory = True
    
    # the dataset name for evaluation
    dataset_name = 'tokyo247'

    # ETR checkpoints path
    ckpt = '/output/checkpoints/epoch13_acc_88_recall@1_84.21_recall@2_91.61_recall@10_93.78.ckpt'

    logger.info("Start evaluate {} datasets", dataset_name)
    query_set = FeatureDataset(data_dir=EVAL_DATASETS[dataset_name]['query_dir'],
                                sample_file=EVAL_DATASETS[dataset_name]['query_path'],
                                max_sequence_len=max_seq_len,
                                gnd_data=EVAL_DATASETS[dataset_name]['gt_path'])
    index_set = FeatureDataset(data_dir=EVAL_DATASETS[dataset_name]['db_dir'],
                                sample_file=EVAL_DATASETS[dataset_name]['db_path'],
                                max_sequence_len=max_seq_len)

    query_loader = DataLoader(query_set, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)
    index_loader = DataLoader(index_set, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)

    print('rerank list', EVAL_DATASETS[dataset_name]['rank_dir'])
    evaluator = Evaluator(dataset_name=dataset_name, 
                    cache_nn_inds=EVAL_DATASETS[dataset_name]['rank_dir'], # 这个代表的就是要被调整的NN文件
                    query_loader=query_loader,
                    index_loader=index_loader,
                    recall=N_RECALLS,
                    topk=TOP_N)

    config = get_cfg_defaults()
    config = lower_config(config)

    device = torch.device('cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu')
    model = ReMatcher(config=config['rmt'])

    print('resume:', ckpt)
    if ckpt is not None:
        # self.rmt = ReMatcher(self.config['rmt'])

        # model = PL_RMT.load_from_checkpoint(ckpt, config=config)

        state_dict = torch.load(ckpt, map_location=torch.device('cpu'))['state_dict']
        # print('keys: ', state_dict.keys(), len(state_dict.keys()))
        state_dict = {k[len('rmt.'):]: v for k, v in state_dict.items()}
        # print('keys: ', state_dict.keys(), len(state_dict.keys()))

        model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    res = evaluator.eval(model)
    print('res: ', res)

    print('\nDone!!!')

if __name__ == "__main__":
    main()
