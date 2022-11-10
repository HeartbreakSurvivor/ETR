import copy
from cv2 import log

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

from ..module.pooling import GeneralizedMeanPooling
from .transformer import TransformerEncoderLayer
from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned

class ReMatcher(nn.Module):
    def __init__(self, config):
        super(ReMatcher, self).__init__()

        assert (config['embed_dim'] % 2 == 0)

        self.embed_dim = config['embed_dim']
        self.layer_names = config['layer_names'] # ['self', 'cross',...]
        self.normalize_before = config['normalize_before']

        if config['pool'] == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif config['pool'] == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1,1))
        elif config['pool'] == 'gem':
            self.pool = GeneralizedMeanPooling(norm=3)
        else:
            raise AttributeError('not support pooling way')

        self.encoder_norm = nn.LayerNorm(self.embed_dim) if self.normalize_before else None
        encoder_layer = TransformerEncoderLayer(d_model=self.embed_dim, 
                                                nhead=config['heads'], 
                                                dim_feedforward=config['ffn_dim'], 
                                                mha_dropout=config['mha_dropout'], 
                                                ffn_dropout=config['ffn_dropout'],
                                                activation=config['activation'], 
                                                normalize_before=self.normalize_before)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        
        # self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # self.pos_encoder = PositionEmbeddingSine(self.embed_dim //2, normalize=True, scale=2.0)
        
        self.remap = nn.Linear(config['global_dim'], self.embed_dim )
        self.seg_encoder = nn.Embedding(2, self.embed_dim )
        self.scale_encoder = nn.Embedding(7, self.embed_dim )

        self.classifier = nn.Linear(self.embed_dim*2, 1) 
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
            src_global, src_local, src_mask, src_scales, src_positions,
            tgt_global, tgt_local, tgt_mask, tgt_scales, tgt_positions,
            normalize=True):

        # src: bsize, slen, fsize (b x n x d)
        # tgt: bsize, slen, fsize (b x n x d)
        src_global = self.remap(src_global) 
        tgt_global = self.remap(tgt_global)
        if normalize:
            src_global = F.normalize(src_global, p=2, dim=-1)
            tgt_global = F.normalize(tgt_global, p=2, dim=-1)
            src_local  = F.normalize(src_local,  p=2, dim=-1)
            tgt_local  = F.normalize(tgt_local,  p=2, dim=-1)
        
        bsize, slen, dim = src_local.size()

        src_global = src_global.unsqueeze(1) + self.seg_encoder(src_local.new_ones((bsize, 1), dtype=torch.long))
        tgt_global = tgt_global.unsqueeze(1) + self.seg_encoder(src_local.new_ones((bsize, 1), dtype=torch.long))
        ##################################################################################################################
        
        ##################################################################################################################
        ## The final model does not use position embeddings for GLD
        src_local = src_local + self.scale_encoder(src_scales) # + self.pos_encoder(src_positions)
        tgt_local = tgt_local + self.scale_encoder(tgt_scales) # + self.pos_encoder(tgt_positions)
        ##################################################################################################################

        ##################################################################################################################
        # segment token,  cls, global_a, local_a, seg, global_b, local_b
        ###################################################################################################################
        src_local = src_local + self.seg_encoder(src_local.new_zeros((bsize, 1), dtype=torch.long))
        tgt_local = tgt_local + self.seg_encoder(src_local.new_zeros((bsize, 1), dtype=torch.long))

        # input_feats = torch.cat([cls_embed, src_global, src_local, sep_embed, tgt_global, tgt_local], 1).permute(1,0,2)
        input_feat_src = torch.cat([src_global, src_local], 1).permute(1,0,2)
        input_feat_tgt = torch.cat([tgt_global, tgt_local], 1).permute(1,0,2)
        
        # input mask
        input_mask_src = torch.cat([
            src_local.new_zeros((bsize, 1), dtype=torch.bool),  
            src_mask 
        ], 1)
        
        input_mask_tgt = torch.cat([
            src_local.new_zeros((bsize, 1), dtype=torch.bool),  
            tgt_mask
        ], 1)

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                # self-attention
                feat_src = layer(src=input_feat_src, src_key_padding_mask=input_mask_src, 
                                 tgt=input_feat_src, tgt_key_padding_mask=input_mask_src)

                feat_tgt = layer(src=input_feat_tgt, src_key_padding_mask=input_mask_tgt, 
                                 tgt=input_feat_tgt, tgt_key_padding_mask=input_mask_tgt)
            elif name == 'cross':
                # cross-attention
                feat_src = layer(src=input_feat_src, src_key_padding_mask=input_mask_src, 
                                 tgt=input_feat_tgt, tgt_key_padding_mask=input_mask_tgt)

                feat_tgt = layer(src=input_feat_tgt, src_key_padding_mask=input_mask_tgt, 
                                 tgt=input_feat_src, tgt_key_padding_mask=input_mask_src)
            else:
                raise KeyError('not support attention layer name')

        if self.encoder_norm is not None:
            feat_src = self.encoder_norm(feat_src)
            feat_tgt = self.encoder_norm(feat_tgt)
        # [501, 12, 128])
        output = torch.cat([feat_src, feat_tgt], dim=-1).permute(1,2,0)
        # [501, 12, 256]) 
        output = rearrange(output, 'b n d -> b n d 1')

        logits = self.pool(output)
        logits = torch.flatten(logits, 1)
        # print('logits', logits.shape)
        # logits = logits[0]
        return self.classifier(logits).view(-1)

