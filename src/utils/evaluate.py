import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
from .metrics import re_match, re_match_sp, mAP_revisited_rerank

class Evaluator(object): 
    def __init__(self, dataset_name,  
                    cache_nn_inds, 
                    query_loader,
                    index_loader,
                    recall=[1,5,10],
                    topk=[100]):

        self.dataset_name = dataset_name
        self.rank_indices = np.load(cache_nn_inds, allow_pickle=True)
        print('eval rank shape', self.rank_indices.shape)

        self.query_loader = query_loader
        self.index_loader = index_loader
        self.recall = recall
        self.topk = topk
        self.load_features()

    def load_features(self):
        print('load val query and db features to memory')
        query_global, query_local, query_mask, query_scales, query_positions = [], [], [], [], []
        index_global, index_local, index_mask, index_scales, index_positions = [], [], [], [], []

        #################### Extracting query features #############################
        for entry in tqdm(self.query_loader, desc='Extracting query features', ncols=80):
            q_global, q_local, q_mask, q_scales, q_positions = entry
            query_global.append(q_global.cpu())
            query_local.append(q_local.cpu())
            query_mask.append(q_mask.cpu())
            query_scales.append(q_scales.cpu())
            query_positions.append(q_positions.cpu())

        self.query_global    = torch.cat(query_global, 0)
        self.query_local     = torch.cat(query_local, 0)
        self.query_mask      = torch.cat(query_mask, 0)
        self.query_scales    = torch.cat(query_scales, 0)
        self.query_positions = torch.cat(query_positions, 0)

        #################### Extracting index features #############################
        for entry in tqdm(self.index_loader, desc='Extracting index features', ncols=80):
            g_global, g_local, g_mask, g_scales, g_positions = entry
            index_global.append(g_global.cpu())
            index_local.append(g_local.cpu())
            index_mask.append(g_mask.cpu())
            index_scales.append(g_scales.cpu())
            index_positions.append(g_positions.cpu())

        self.db_global    = torch.cat(index_global, 0)
        self.db_local     = torch.cat(index_local, 0)
        self.db_mask      = torch.cat(index_mask, 0)
        self.db_scales    = torch.cat(index_scales, 0)
        self.db_positions = torch.cat(index_positions, 0)

        print("query_global shape", self.query_global.shape)
        print("index_global shape", self.db_global.shape)

    def eval(self, model):
        if self.dataset_name in ('rparis6k', 'roxford5k'):
            metrics = mAP_revisited_rerank(model=model,
                            cache_nn_inds=self.rank_indices,
                            query_global=self.query_global, query_local=self.query_local, query_mask=self.query_mask, query_scales=self.query_scales, query_positions=self.query_positions, 
                            gallery_global=self.db_global, gallery_local=self.db_local, gallery_mask=self.db_mask, gallery_scales=self.db_scales, gallery_positions=self.db_positions, 
                            ks=self.recall,
                            topk_list=self.topk,
                            gnd_file=self.query_loader.dataset.gnd_data
                            )
        else:
            metrics = re_match(
                self.dataset_name,
                model=model, 
                cache_nn_inds=self.rank_indices,
                # query and db
                query_global=self.query_global, query_local=self.query_local, query_mask=self.query_mask, query_scales=self.query_scales, query_positions=self.query_positions, 
                index_global=self.db_global, index_local=self.db_local, index_mask=self.db_mask, index_scales=self.db_scales, index_positions=self.db_positions, 
                ks=self.recall,
                top_k=self.topk,
                gnd_file=self.query_loader.dataset.gnd_data,
            )
        return metrics


class Evaluator_sp(object): 
    def __init__(self, dataset_name,  
                    cache_nn_inds, 
                    query_loader,
                    index_loader,
                    recall=[1,5,10],
                    topk=[100]):

        self.dataset_name = dataset_name
        self.rank_indices = np.load(cache_nn_inds, allow_pickle=True)
        print('eval rank shape', self.rank_indices.shape)

        self.query_loader = query_loader
        self.index_loader = index_loader
        self.recall = recall
        self.topk = topk
        self.load_features()

    def load_features(self):
        print('load val query and db features to memory')
        query_global, query_local, query_mask, query_positions = [], [], [], []
        index_global, index_local, index_mask, index_positions = [], [], [], []

        #################### Extracting query features #############################
        for entry in tqdm(self.query_loader, desc='Extracting query features', ncols=80):
            q_global, q_local, q_mask, q_positions = entry
            query_global.append(q_global.cpu())
            query_local.append(q_local.cpu())
            query_mask.append(q_mask.cpu())
            query_positions.append(q_positions.cpu())

        self.query_global    = torch.cat(query_global, 0)
        self.query_local     = torch.cat(query_local, 0)
        self.query_mask      = torch.cat(query_mask, 0)
        self.query_positions = torch.cat(query_positions, 0)

        #################### Extracting index features #############################
        for entry in tqdm(self.index_loader, desc='Extracting index features', ncols=80):
            g_global, g_local, g_mask, g_positions = entry
            index_global.append(g_global.cpu())
            index_local.append(g_local.cpu())
            index_mask.append(g_mask.cpu())
            index_positions.append(g_positions.cpu())

        self.db_global    = torch.cat(index_global, 0)
        self.db_local     = torch.cat(index_local, 0)
        self.db_mask      = torch.cat(index_mask, 0)
        self.db_positions = torch.cat(index_positions, 0)

        print("query_global shape", self.query_global.shape)
        print("index_global shape", self.db_global.shape)

    def eval(self, model):
        # 只需要传入model

        if self.dataset_name in ('rparis6k', 'roxford5k'):
            pass
            # metrics = mAP_revisited_rerank(model=model,
            #                 cache_nn_inds=self.rank_indices,
            #                 query_global=self.query_global, query_local=self.query_local, query_mask=self.query_mask, query_scales=self.query_scales, query_positions=self.query_positions, 
            #                 gallery_global=self.db_global, gallery_local=self.db_local, gallery_mask=self.db_mask, gallery_scales=self.db_scales, gallery_positions=self.db_positions, 
            #                 ks=self.recall,
            #                 topk_list=self.topk,
            #                 gnd_file=self.query_loader.dataset.gnd_data
            #                 )
        else:
            metrics = re_match_sp(
                self.dataset_name,
                model=model, 
                cache_nn_inds=self.rank_indices,
                # query and db
                query_global=self.query_global, query_local=self.query_local, query_mask=self.query_mask, query_positions=self.query_positions, 
                index_global=self.db_global, index_local=self.db_local, index_mask=self.db_mask, index_positions=self.db_positions, 
                ks=self.recall,
                top_k=self.topk,
                gnd_file=self.query_loader.dataset.gnd_data,
            )
        return metrics

