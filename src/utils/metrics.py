import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List

import pickle
import time
from loguru import logger
from .revisited_op import compute_metrics

def compute_recall(predictions, gt, numQ, n_values, recall_str=''):
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / numQ
    logger.info('correct_at_n: {}, numQ: {}', recall_at_n, numQ)

    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        # print("====> Recall {}@{}: {:.4f}".format(recall_str, n, recall_at_n[i]))
        logger.info("====> Recall {}@{}: {:.4f}", recall_str, n, recall_at_n[i])
    return all_recalls

@torch.no_grad()
def re_match(
    dataset_name: str,
    model: nn.Module,
    cache_nn_inds: torch.Tensor,

    query_global: torch.Tensor, query_local: torch.Tensor, query_mask: torch.Tensor, query_scales: torch.Tensor, query_positions: torch.Tensor,
    index_global: torch.Tensor, index_local: torch.Tensor, index_mask: torch.Tensor, index_scales: torch.Tensor, index_positions: torch.Tensor,
    ks: List[int],
    top_k, # rerank for topk
    gnd_file):

    device = next(model.parameters()).device
    print('current device:', device)

    # query_global    = query_global.to(device)
    # query_local     = query_local.to(device)
    # query_mask      = query_mask.to(device)
    # query_scales    = query_scales.to(device)
    # query_positions = query_positions.to(device)

    num_samples, pred = cache_nn_inds.shape
    logger.info('num_samples, pred: {}, {}', num_samples, pred)
    cache_nn_inds = torch.from_numpy(cache_nn_inds).to(device)

    # top_k = max(10, top_k)
    # top_k = [10, 50, 100, 200, 300]
    # top_k = [100]
    logger.info('top_k: {}', top_k)

    recalls = {}
    for tk in top_k:
        logger.info('current topk {}', tk)

        rerank_topk = cache_nn_inds[:, :tk]
        medium_nn_inds = deepcopy(rerank_topk.cpu().data.numpy())
        medium_nn_inds = torch.from_numpy(medium_nn_inds)
        
        scores = []
        total_time = 0.0
        for i in tqdm(range(num_samples), desc='rerank...', ncols=80): 
            nnids = cache_nn_inds[i,:tk]
            # logger.info('nnids shaoe', nnids.shape)

            tgt_global    = index_global[nnids]
            tgt_local     = index_local[nnids]
            tgt_mask      = index_mask[nnids]
            tgt_scales    = index_scales[nnids]
            tgt_positions = index_positions[nnids]

            src_global    = query_global[i].unsqueeze(0).repeat(tk, 1)
            src_local     = query_local[i].unsqueeze(0).repeat(tk, 1, 1)
            src_mask      = query_mask[i].unsqueeze(0).repeat(tk, 1)
            src_scales    = query_scales[i].unsqueeze(0).repeat(tk, 1)
            src_positions = query_positions[i].unsqueeze(0).repeat(tk, 1, 1)

            # logger.info('tgt_global shaoe', tgt_global.shape)
            # logger.info('src_global shaoe', src_global.shape)

            src_global    = src_global.to(device)
            src_local     = src_local.to(device)
            src_mask      = src_mask.to(device)
            src_scales    = src_scales.to(device)
            src_positions = src_positions.to(device)

            tgt_global = tgt_global.to(device)
            tgt_local = tgt_local.to(device)
            tgt_mask = tgt_mask.to(device)
            tgt_scales = tgt_scales.to(device)
            tgt_positions = tgt_positions.to(device)

            start = time.time()
            # inference
            current_scores = model(
                src_global, src_local, src_mask, src_scales, src_positions,
                tgt_global, tgt_local, tgt_mask, tgt_scales, tgt_positions
            )
            end = time.time()
            if i % 500 == 0:
                logger.info('one shot to rerank top100 costs {} ', end - start)

            total_time += end - start
            # logger.info('current_scores', current_scores.shape)
            scores.append(current_scores.cpu().data)

        scores = torch.stack(scores, 0).cpu()
        logger.info('scores {} ', scores.shape)
        logger.info('total time: {}', total_time)
        logger.info('total samples: {}', num_samples)
        logger.info('time per sample: {}', total_time / num_samples)

        _, indices = torch.sort(scores, dim=-1, descending=True)
        # np.save('indices', indices)
        logger.info('indices: {}', indices.shape)
        logger.info('indices[0]: {}', indices[0][:20])

        closest_indices = medium_nn_inds.gather(-1, indices)

        ranks = deepcopy(medium_nn_inds)
        ranks[:, :tk] = deepcopy(closest_indices)

        # np.save('scores_rerank', scores)

        # ranks = ranks.cpu().data.numpy().T
        ranks = ranks.cpu().data.numpy()
        logger.info('ranks: {}', ranks.shape)

        if not gnd_file or dataset_name in ('robotcarv2', 'aachen', 'aachen_v1'):
            logger.info('evaluate robotcarV2 or aachen done!')
        else:
            gnd = np.load(gnd_file, allow_pickle=True)
            re = compute_recall(ranks, gnd, num_samples, ks, recall_str='rmt')
            recalls[tk] = re

    logger.info('Re-rank done.')
    return recalls

@torch.no_grad()
def re_match_sp(
    dataset_name: str,
    model: nn.Module,
    cache_nn_inds: torch.Tensor,

    query_global: torch.Tensor, query_local: torch.Tensor, query_mask: torch.Tensor, query_positions: torch.Tensor,
    index_global: torch.Tensor, index_local: torch.Tensor, index_mask: torch.Tensor, index_positions: torch.Tensor,
    ks: List[int],
    top_k,
    gnd_file):

    device = next(model.parameters()).device
    print('current device:', device)

    # 这里先放在内存中，全放到GPU，估计放不下，
    # query_global    = query_global.to(device)
    # query_local     = query_local.to(device)
    # query_mask      = query_mask.to(device)
    # query_scales    = query_scales.to(device)
    # query_positions = query_positions.to(device)

    # 其实整个模型负责处理是rerank部分，即已经存在的检索
    num_samples, pred = cache_nn_inds.shape
    logger.info('num_samples, pred: {}, {}', num_samples, pred)
    cache_nn_inds = torch.from_numpy(cache_nn_inds).to(device)

    # 最小也得对 top10 进行rerank
    # top_k = max(10, top_k)
    # top_k = [10, 50, 100, 200, 300]
    # top_k = [100]
    logger.info('top_k: {}', top_k)

    recalls = {}
    for tk in top_k:
        logger.info('current topk {}', tk)

        # 取前topk个预测的值进行重排
        rerank_topk = cache_nn_inds[:, :tk]
        medium_nn_inds = deepcopy(rerank_topk.cpu().data.numpy())
        medium_nn_inds = torch.from_numpy(medium_nn_inds)
        
        scores = []
        total_time = 0.0
        # 外层循环是num_samples, 即 query image 的数量
        for i in tqdm(range(num_samples), desc='rerank...', ncols=80): 
            # 取出当前query，对应需要rerank的 top_k个
            nnids = cache_nn_inds[i,:tk]
            # logger.info('nnids shaoe', nnids.shape)

            # 取出这些预测的db图片的下标
            tgt_global    = index_global[nnids]
            tgt_local     = index_local[nnids]
            tgt_mask      = index_mask[nnids]
            tgt_positions = index_positions[nnids]

            # 把 query重复 topk次，以便跟db拼接起来
            src_global    = query_global[i].unsqueeze(0).repeat(tk, 1)
            src_local     = query_local[i].unsqueeze(0).repeat(tk, 1, 1)
            src_mask      = query_mask[i].unsqueeze(0).repeat(tk, 1)
            src_positions = query_positions[i].unsqueeze(0).repeat(tk, 1, 1)

            # logger.info('tgt_global shaoe', tgt_global.shape)
            # logger.info('src_global shaoe', src_global.shape)

            # 这里可以勾选起来
            src_global    = src_global.to(device)
            src_local     = src_local.to(device)
            src_mask      = src_mask.to(device)
            src_positions = src_positions.to(device)

            tgt_global = tgt_global.to(device)
            tgt_local = tgt_local.to(device)
            tgt_mask = tgt_mask.to(device)
            tgt_positions = tgt_positions.to(device)

            start = time.time()
            # 模型inference
            current_scores = model(
                src_global, src_local, src_mask, src_positions,
                tgt_global, tgt_local, tgt_mask, tgt_positions
            )
            end = time.time()
            if i % 500 == 0:
                logger.info('one shot to rerank top100 costs {} ', end - start)

            total_time += end - start
            # logger.info('current_scores', current_scores.shape)
            # 获得当前query图片的候选集合重排之后的得分
            scores.append(current_scores.cpu().data)

        scores = torch.stack(scores, 0).cpu()
        logger.info('scores {} ', scores.shape)
        logger.info('total time: {}', total_time)
        logger.info('total samples: {}', num_samples)
        logger.info('time per sample: {}', total_time / num_samples)

        # 按照得分降序排列，输出的分数越大代表越像，
        _, indices = torch.sort(scores, dim=-1, descending=True)
        # np.save('indices', indices)
        logger.info('indices: {}', indices.shape)
        # logger.info('indices[0]: {}', indices[0][:20])

        closest_indices = medium_nn_inds.gather(-1, indices)

        ranks = deepcopy(medium_nn_inds)
        ranks[:, :tk] = deepcopy(closest_indices)

        # np.save('scores_rerank', scores)

        # ranks = ranks.cpu().data.numpy().T
        ranks = ranks.cpu().data.numpy()
        logger.info('ranks: {}', ranks.shape)

        # if tk==100:
            # np.save('/root/halo/pictures/ranks/' + dataset_name + '/top_' + str(tk) + '_rank', ranks)
        # np.save('/root/halo/pictures/ranks/pitts30k/netvlad/rmt_rank_top{}_index'.format(tk), ranks)
        # np.save('/root/halo/pictures/ranks/pitts30k/delg/rmt_rank_top{}_index'.format(tk), ranks)

        # d_name = 'kampala'
        # np.save('/root/halo/code/MSFFT/datasets/MSLS_test/{}/delg_feats/rmt_{}_top{}_ranks'.format(dataset_name, dataset_name, tk), ranks)
        # np.save('/root/halo/code/MSFFT/datasets/MSLS_test/{}/sp_feats/rmt_sp_v7_top{}_ranks'.format(dataset_name, tk), ranks)
        # np.save('/root/halo/pictures/ranks/MSLS val/cph/GETR_top_' + str(tk) + '_rank', ranks)
        # np.save('/root/halo/pictures/ranks/pitts30k/apgem/GETR_top_' + str(tk) + '_rank', ranks)
        
        if not gnd_file or dataset_name in ('robotcarv2', 'aachen', 'aachen_v1'):
            logger.info('evaluate robotcarV2 or aachen done!')
            # np.save('/root/halo/code/MSFFT/datasets/Aachen_v1.1/rmt_ranks/rmt_rank_top{}_index'.format(tk), ranks)
        else:
            gnd = np.load(gnd_file, allow_pickle=True)
            re = compute_recall(ranks, gnd, num_samples, ks, recall_str='rmt')
            recalls[tk] = re

    logger.info('Re-rank done.')
    return recalls


@torch.no_grad()
def mAP_revisited_rerank(
    model: nn.Module,
    cache_nn_inds: torch.Tensor,

    query_global: torch.Tensor, query_local: torch.Tensor, query_mask: torch.Tensor, query_scales: torch.Tensor, query_positions: torch.Tensor,
    gallery_global: torch.Tensor, gallery_local: torch.Tensor, gallery_mask: torch.Tensor, gallery_scales: torch.Tensor, gallery_positions: torch.Tensor,
    ks: List[int],
    topk_list, # rerank topk candidates
    gnd_file) -> Dict[str, float]:

    device = next(model.parameters()).device
    query_global    = query_global.to(device)
    query_local     = query_local.to(device)
    query_mask      = query_mask.to(device)
    query_scales    = query_scales.to(device)
    query_positions = query_positions.to(device)

    cache_nn_inds = torch.from_numpy(cache_nn_inds)
    num_samples, top_k = cache_nn_inds.size()
    print('num_samples, top_k', num_samples, top_k)
    # top_k = min(100, top_k)

    with open(gnd_file, 'rb') as f:
        gnd = pickle.load(f)
    print('gnd', type(gnd))

    result = {}
    for top_k in topk_list:
        ########################################################################################
        ## Medium
        ########################################################################################
        medium_nn_inds = deepcopy(cache_nn_inds.cpu().data.numpy())

        # Exclude the junk images as in DELG (https://github.com/tensorflow/models/blob/44cad43aadff9dd12b00d4526830f7ea0796c047/research/delf/delf/python/detect_to_retrieve/image_reranking.py#L190)
        for i in range(num_samples):
            junk_ids = gnd['gnd'][i]['junk']
            all_ids = medium_nn_inds[i]
            pos = np.in1d(all_ids, junk_ids)
            neg = np.array([not x for x in pos])
            new_ids = np.concatenate([np.arange(len(all_ids))[neg], np.arange(len(all_ids))[pos]])
            new_ids = all_ids[new_ids]
            medium_nn_inds[i] = new_ids
        medium_nn_inds = torch.from_numpy(medium_nn_inds)
        
        scores = []
        for i in tqdm(range(top_k)):
            nnids = medium_nn_inds[:, i]
            index_global    = gallery_global[nnids]
            index_local     = gallery_local[nnids]
            index_mask      = gallery_mask[nnids]
            index_scales    = gallery_scales[nnids]
            index_positions = gallery_positions[nnids]
            current_scores = model(
                query_global, query_local, query_mask, query_scales, query_positions,
                index_global.to(device),
                index_local.to(device),
                index_mask.to(device),
                index_scales.to(device),
                index_positions.to(device))

            scores.append(current_scores.cpu().data)

        scores = torch.stack(scores, -1) # 70 x 100
        closest_dists, indices = torch.sort(scores, dim=-1, descending=True)
        closest_indices = torch.gather(medium_nn_inds, -1, indices)
        ranks = deepcopy(medium_nn_inds)
        ranks[:, :top_k] = deepcopy(closest_indices)
        ranks = ranks.cpu().data.numpy().T
        # pickle_save('medium_nn_inds.pkl', ranks.T)
        medium = compute_metrics('revisited', ranks, gnd['gnd'], kappas=ks)

        ########################################################################################
        ## Hard
        ########################################################################################

        hard_nn_inds = deepcopy(cache_nn_inds.cpu().data.numpy())
        # Exclude the junk images as in DELG (https://github.com/tensorflow/models/blob/44cad43aadff9dd12b00d4526830f7ea0796c047/research/delf/delf/python/detect_to_retrieve/image_reranking.py#L190)
        for i in range(num_samples):
            junk_ids = gnd['gnd'][i]['junk'] + gnd['gnd'][i]['easy']
            all_ids = hard_nn_inds[i]
            pos = np.in1d(all_ids, junk_ids)
            neg = np.array([not x for x in pos])
            new_ids = np.concatenate([np.arange(len(all_ids))[neg], np.arange(len(all_ids))[pos]])
            new_ids = all_ids[new_ids]
            hard_nn_inds[i] = new_ids
        hard_nn_inds = torch.from_numpy(hard_nn_inds)

        scores = []
        for i in tqdm(range(top_k)):
            nnids = hard_nn_inds[:, i]
            index_global    = gallery_global[nnids]
            index_local     = gallery_local[nnids]
            index_mask      = gallery_mask[nnids]
            index_scales    = gallery_scales[nnids]
            index_positions = gallery_positions[nnids]
            current_scores = model(
                query_global, query_local, query_mask, query_scales, query_positions,
                index_global.to(device),
                index_local.to(device),
                index_mask.to(device),
                index_scales.to(device),
                index_positions.to(device))
            scores.append(current_scores.cpu().data)
        scores = torch.stack(scores, -1) # 70 x 100
        closest_dists, indices = torch.sort(scores, dim=-1, descending=True)
        closest_indices = torch.gather(hard_nn_inds, -1, indices)

        # pickle_save('nn_inds_rerank.pkl', closest_indices)
        # pickle_save('nn_dists_rerank.pkl', closest_dists)

        ranks = deepcopy(hard_nn_inds)
        ranks[:, :top_k] = deepcopy(closest_indices)
        ranks = ranks.cpu().data.numpy().T
        # pickle_save('hard_nn_inds.pkl', ranks.T)
        hard = compute_metrics('revisited', ranks, gnd['gnd'], kappas=ks)

        ########################################################################################  
        out = {
            'M_map': float(medium['M_map']), 
            'H_map': float(hard['H_map']),
            'M_mp':  medium['M_mp'].tolist(),
            'H_mp': hard['H_mp'].tolist(),
        }
        result[top_k] = out
        
    # json_save('eval_revisited.json', out)
    return result

