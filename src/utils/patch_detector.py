from enum import unique
from typing import Dict
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

# from ..model.module.pooling import GeM

INF = 1e9

class PatchDetector():
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        self.thr = 0.01 # config['threshold']
        self.temperature = config['temperature']
        self.feat_c_h = config['transformer_coarse']['image_size'][0]
        self.feat_c_w = config['transformer_coarse']['image_size'][1]
        self.scale_h = config['img_h'] // self.feat_c_h
        self.scale_w = config['img_w'] // self.feat_c_w

        self.patch_scale_type = config['patch_scale_type']
        self.multi_scale_type = config['multi_scale_type']
        self.patch_detect_type = config['patch_detect_type']
        self.cross_patch_scale_type = config['cross_patch_scale_type']

        # self.pool = GeM()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cooridinate = self.gen_cooridinate()
        
        self.cooridinate = torch.Tensor(self.cooridinate).float().to(device)

    def gen_cooridinate(self):
        co = []
        # cooridinate offset to center of patches
        dx, dy = self.scale_h // 2, self.scale_w // 2
        for i in range(self.feat_c_h):
            for j in range(self.feat_c_w):
                x, y = i*self.scale_h + dx, j*self.scale_w + dy
                co.append((x, y))
        # print('co: ', co, len(co))
        return np.array(co)

    def detect(self, feat_a, feat_b):
        """
        feat_a: Dict {'feat_c', 'feat_f0', 'feat_f1'}
        feat_b: Dict {'feat_c', 'feat_f0', 'feat_f1'}
        """
        return self.key_patch_detect(feat_a, feat_b)

    def get_feat_map(self, feat_a, feat_b):
        if self.patch_scale_type == 'single_c':
            fa, fb = feat_a['feat_c'], feat_b['feat_c']
        elif self.patch_scale_type == 'single_f0':
            fa, fb = feat_a['feat_f0'], feat_b['feat_f0']
        elif self.patch_scale_type == 'single_f1':
            fa, fb = feat_a['feat_f1'], feat_b['feat_f1']
        elif self.patch_scale_type == 'multi_scale': # multi-scale feature fusion
            # concatneate multi-scale patch embedding
            fa = torch.cat((feat_a['feat_c'], feat_a['feat_f0'], feat_a['feat_f1']), dim=-1)
            fb = torch.cat((feat_b['feat_c'], feat_b['feat_f0'], feat_b['feat_f1']), dim=-1)
        else:
            raise NotImplementedError('scale type must in [''single_c', 'single_f0', 'single_f1', 'multi_scale'']')
        return fa, fb
    
    def get_multi_scale_feat_map(self, feat_a, feat_b):
        f0_size = feat_a['feat_f0'].shape[-1] 

        if self.multi_scale_type in ['c_f0', 'f0_f1', 'c_f1']:
            ca, cb = self.multi_scale_type.split('_')
            ca, cb = 'feat' + ca, 'feat' + cb
            print('ca, cb', ca, cb)
            fa, fb = feat_a[ca], feat_b[cb]

            print('fa fb', fa.shape, fb.shape)
            if fa.shape[-1] != f0_size:
                fa = F.interpolate(fb, size=f0_size, mode='linear', align_corners=True)
            if fb.shape[-1] != f0_size:
                fb = F.interpolate(fb, size=f0_size, mode='linear', align_corners=True)
            return fa, fb
        else:
            raise NotImplementedError('scale type must in [''c_f0', 'f0_f1', 'c_f1'']')

    def get_cross_multi_scale_feat_map(self, feat_a, feat_b):
        f0_size = feat_a['feat_f0'].shape[-1] 

        if self.cross_patch_scale_type in ['c_c', 'f0_f0', 'f1_f1']:
            ca, cb = self.cross_patch_scale_type.split('_')
            ca, cb = 'feat_' + ca, 'feat_' + cb 
            fa, fb = feat_a[ca], feat_b[cb]
        elif self.cross_patch_scale_type == 'ms_ms':
            fa = torch.cat((feat_a['feat_c'], feat_a['feat_f0'], feat_a['feat_f1']), dim=-1)
            fb = torch.cat((feat_b['feat_c'], feat_b['feat_f0'], feat_b['feat_f1']), dim=-1)
        elif self.cross_patch_scale_type in ['f0_c', 'f0_f1', 'c_f0', 'f1_f0', 'c_f1', 'f1_c']:
            ca, cb = self.cross_patch_scale_type.split('_')
            ca, cb = 'feat_' + ca, 'feat_' + cb
            fa, fb = feat_a[ca], feat_b[cb]
            if fa.shape[-1] != f0_size:
                fa = F.interpolate(fb, size=f0_size, mode='linear', align_corners=True)
            if fb.shape[-1] != f0_size:
                fb = F.interpolate(fb, size=f0_size, mode='linear', align_corners=True)
        return fa, fb

    def key_patch_detect(self, feat_a, feat_b):
        if self.patch_detect_type == 'patch_attention':
            # 1. get feature map for imageA and imageB
            fa, fb = self.get_feat_map(feat_a, feat_b)
            kpts_a, desc_a = self.single_patch_attention(fa)
            kpts_b, desc_b = self.single_patch_attention(fb)
            return kpts_a, desc_a, kpts_b, desc_b
        elif self.patch_detect_type == 'self_attention':
            fa, fb = self.get_feat_map(feat_a, feat_b)
            kpts_a = self.self_patch_attention(fa, fa)
            kpts_b = self.self_patch_attention(fb, fb)
            return kpts_a, kpts_b
        elif self.patch_detect_type == 'multi_scale_attention':
            # cross attention 需要不同尺度的特征图的patch 的维度也一致
            # 但是这里是不一致的，简单点可以通过线性插值的方式来转换成相同的size
            fa, fb = self.get_multi_scale_feat_map(feat_a, feat_b)
            # 计算坐标
            return self.cross_patch_attention(fa, fb)
        elif self.patch_detect_type == 'cross_patch_attention':
            fa, fb = self.get_cross_multi_scale_feat_map(feat_a, feat_b)
            return self.cross_patch_attention(fa, fb)
            # post process to get kpts
        else:
            raise NotImplementedError('match type must in [''patch_attention'', ''self_attention'', ''multi_scale_attention'', ''cross_patch_attention'']')

    def gen_keypatch_mask(self, patch):
    # def key_patch_filter(self, patch):
        o = rearrange(patch, 'p d -> p d 1')
        print('o shape', o.shape)
        o = self.pool(o)
        print('o shape', o.shape)
        o = torch.flatten(o, 1)
        print('o shape', o.shape)
        o = F.softmax(o, dim=-1) # 对patch那一维进行softmax

        mask = patch > self.thr
        print('mask', mask.shape)
        return mask
    
    def key_patch_detection(self, mask):
        # first version
        # res = []
        # for p, m in zip(patch, mask):
        #     desc = []
        #     keypatch = []
        #     for idx, mi in enumerate(m):
        #         if mi == True: # key patch
        #             # get center coordinate of patch by patch index
        #             kpts = self.cooridinate(idx)
        #             keypatch.append(kpts)
        #             desc.append(p[idx])
        #     res.append((keypatch, desc))
        # res = torch.stack(res, dim=1)

        # 找到所有部位False的patch的下标
        idxs = torch.nonzero(mask)
        idxs = idxs.T
        print('idxs', idxs, idxs.shape)
        if not idxs:
            return None, None
        # b_idxs, kpts = torch.chunk(idxs, 2, dim=0)
        # b_idxs, kpts = idxs[0], idxs[1]
        kpts = idxs[0]

        kpts = self.cooridinate[kpts]
        # print('b_idxs, kpts', b_idxs, kpts)
        # 使用的时候，根据batch序号来跟b_idxs取一个mask，然后去取对应的关键点坐标
        # return b_idxs, kpts
        return kpts

    def single_patch_attention(self, feat):
        # 1. generate key patch mask
        # o = rearrange(feat, 'b p d -> b p d 1')
        print('feat', feat)
        print('pool', self.pool)
        o = rearrange(feat, 'p d -> p d 1')
        print('o shape', o.shape)
        o = self.pool(o)
        print(o)
        print('o shape', o.shape)
        # o = torch.flatten(o, 1)
        o = torch.squeeze(o)
        print('o shape', o.shape)
        o = F.softmax(o, dim=-1) # 对patch那一维进行softmax
        print('softmax:', o)

        mask = o > self.thr
        # 2. get key patch center cooridinate
        return self.key_patch_detection(mask)

    def self_patch_attention(self, feat1, feat2):
        # normalize
        feat1 = feat1 / (feat1.shape[-1]**.5)
        feat2 = feat2 / (feat2.shape[-1]**.5)
        print('feat1', feat1)
        print('feat2', feat2)
        # 1. calculate self-attention matrix
        sim_matrix = (feat1 @ feat2.T) / 0.01
        # sim_matrix = torch.einsum("nc, pc -> np", feat1, feat2) / self.temperature
        # sim_matrix = torch.einsum("bnc, bpc -> bnp", feat1, feat2) / self.temperature
        
        # 2. use mask to prevent patch compare with itself, which will produce high similarity
        # 一个patch self-attention的时候，需要mask，不同特征图的时候，不需要mask
        sim_matrix.masked_fill_(torch.eye(sim_matrix.shape[-1]).bool(), -INF)

        # dual-softmax
        a = F.softmax(sim_matrix, 0)
        print('sum col: ', a[:,0].sum())

        b = F.softmax(sim_matrix, 1)
        conf_matrix = a * b
        print('sum row: ', b[0].sum())
        # conf_matrix = F.softmax(sim_matrix, 0) * F.softmax(sim_matrix, 1)
        print('conf_m: ', conf_matrix)
        # 3. filter patch similarity that less than threshold
        mask = conf_matrix > 0
        print('mask: ', mask)

        # 4. find two patches while are mutual nearest neighbor
        mask = mask \
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0]) \
            * (conf_matrix == conf_matrix.max(dim=0, keepdim=True)[0])
        print('mask: ', mask, mask.shape)
        # print(mask[0][0])

        # 5. find all valid patch matches
        # this only works when at most one `True` in each row
        print('找到所有可能的粗粒度匹配点')
        mask_v, all_j_idxs = mask.max(dim=1)
        print("mask_v, all_j_idxs: ", mask_v.shape, all_j_idxs.shape)
        print("mask_v: ", mask_v)
        print("all_j_idxs: ", all_j_idxs)

        i_idxs = torch.where(mask_v) # return a tuple
        i_idxs = i_idxs[0] 
        # print("b_idxs, i_idxs: ", b_idxs.shape, i_idxs.shape)
        # print("b_idxs", b_idxs)
        print("i_idxs", i_idxs)
        j_idxs = all_j_idxs[i_idxs]

        # assert i_idxs.shape == j_idxs.shape

        print('j_ids: ', j_idxs)
        print('matched key-patches: ', j_idxs, j_idxs.shape)
        # get key patch center cooridinate
        kpts = self.cooridinate[i_idxs]
        print('b_idxs, kpts', kpts)
        # 使用的时候，根据batch序号来跟b_idxs取一个mask，然后去取对应的关键点坐标
        return kpts

    def cross_multi_attention(self, feat1, feat2):
        return self.cross_patch_attention(feat1, feat2)

    def cross_patch_attention(self, feat1, feat2):
        # normalize
        feat1 = feat1 / (feat1.shape[-1]**.5)
        feat2 = feat2 / (feat2.shape[-1]**.5)
        # print('feat1', feat1)
        # print('feat2', feat2)
        # 1. calculate self-attention matrix
        sim_matrix = (feat1 @ feat2.T) / 0.01 # self.temperature
        # 2. use mask to prevent patch compare with itself, which will produce high similarity
        # 一个patch self-attention的时候，需要mask，不同特征图的时候，不需要mask
        # sim_matrix.masked_fill_(torch.eye(sim_matrix.shape[-1]).bool(), -INF)

        # dual-softmax
        conf_matrix = F.softmax(sim_matrix, 0) * F.softmax(sim_matrix, 1)
        # print('conf_m: ', conf_matrix)
        # 3. filter patch similarity that less than threshold
        mask = conf_matrix > 0.0 # self.thr
        # print('thr mask: \n', mask)

        # 4. find two patches while are mutual nearest neighbor
        mask = mask \
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0]) \
            * (conf_matrix == conf_matrix.max(dim=0, keepdim=True)[0])
        # print('mutual mask: \n', mask)
        # print(mask[0][0])

        # 5. find all valid patch matches
        # this only works when at most one `True` in each row
        # print('找到所有可能的粗粒度匹配点')
        mask_v, all_j_idxs = mask.max(dim=1)
        # print("mask_v, all_j_idxs: ", mask_v.shape, all_j_idxs.shape)
        # print("mask_v: ", mask_v)
        # print("all_j_idxs: ", all_j_idxs)

        i_idxs = torch.where(mask_v) # return a tuple
        i_idxs = i_idxs[0]
        # print("i_idxs: ", i_idxs)
        j_idxs = all_j_idxs[i_idxs]

        # assert i_idxs.shape == j_idxs.shape

        # print("i_idxs: ", i_idxs, len(i_idxs))
        # print('j_idxs: ', j_idxs, len(j_idxs))

        # print('matched key-patches: ', len(j_idxs))
        # get key patch center cooridinate
        kpts_a = self.cooridinate[i_idxs]
        kpts_b = self.cooridinate[j_idxs]
        # print('kpts_a', kpts_a)
        # print('kpts_b', kpts_b)
        # 使用的时候，根据batch序号来跟b_idxs取一个mask，然后去取对应的关键点坐标
        return kpts_a, kpts_b

if __name__ == '__main__':
    def cross_patch_attention(feat1, feat2, co):
        # normalize
        feat1 = feat1 / (feat1.shape[-1]**.5)
        feat2 = feat2 / (feat2.shape[-1]**.5)
        print('feat1', feat1)
        print('feat2', feat2)
        # 1. calculate self-attention matrix
        sim_matrix = (feat1 @ feat2.T) / 0.01
        # sim_matrix = torch.einsum("nc, pc -> np", feat1, feat2) / self.temperature
        # sim_matrix = torch.einsum("bnc, bpc -> bnp", feat1, feat2) / self.temperature
        
        # 2. use mask to prevent patch compare with itself, which will produce high similarity
        # 一个patch self-attention的时候，需要mask，不同特征图的时候，不需要mask
        # sim_matrix.masked_fill_(torch.eye(sim_matrix.shape[-1]).bool(), -INF)

        # dual-softmax
        # a = F.softmax(sim_matrix, 0)
        # print('sum col: ', a[:,0].sum())
        # b = F.softmax(sim_matrix, 1)
        # conf_matrix = a * b
        # print('sum row: ', b[0].sum())
        conf_matrix = F.softmax(sim_matrix, 0) * F.softmax(sim_matrix, 1)
        print('conf_m: ', conf_matrix)
        # 3. filter patch similarity that less than threshold
        mask = conf_matrix > 0.1
        print('thr mask: \n', mask)

        # 4. find two patches while are mutual nearest neighbor
        mask = mask \
            * (conf_matrix == conf_matrix.max(dim=0, keepdim=True)[0]) \
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
        print('mutual mask: \n', mask)
        # print(mask[0][0])

        # 5. find all valid patch matches
        # this only works when at most one `True` in each row
        print('找到所有可能的粗粒度匹配点')
        mask_v, all_j_idxs = mask.max(dim=1)
        print("mask_v, all_j_idxs: ", mask_v.shape, all_j_idxs.shape)
        print("mask_v: ", mask_v)
        print("all_j_idxs: ", all_j_idxs)

        i_idxs = torch.where(mask_v) # return a tuple
        i_idxs = i_idxs[0]
        print("i_idxs: ", i_idxs)
        j_idxs = all_j_idxs[i_idxs]

        # assert i_idxs.shape == j_idxs.shape

        print("i_idxs: ", i_idxs, len(i_idxs))
        print('j_idxs: ', j_idxs, len(j_idxs))

        print('matched key-patches: ', len(j_idxs))
        # get key patch center cooridinate
        kpts_a = co[i_idxs]
        kpts_b = co[j_idxs]
        # print('kpts_a', kpts_a)
        # print('kpts_b', kpts_b)
        # 使用的时候，根据batch序号来跟b_idxs取一个mask，然后去取对应的关键点坐标
        return kpts_a, kpts_b

    def gen_cooridinate():
        co = []
        scale_h = 32
        scale_w = 32
        # cooridinate offset to center of patches
        dx, dy = scale_h // 2, scale_w // 2
        for i in range(15):
            for j in range(20):
                x, y = i*scale_h + dx, j*scale_w + dy
                co.append((x, y))
        # print('co: ', co, len(co))
        return torch.Tensor(co).float()

    co = gen_cooridinate()

    a = torch.randn(5, 128)
    
    b = torch.randn(5, 128)
    print(a)
    print(b)
    ka, kb = cross_patch_attention(a, b, co)
    # print(ka, kb)
    # m = m * (c == c.max(dim=1, keepdim=True)[0])* (c == c.max(dim=0, keepdim=True)[0])