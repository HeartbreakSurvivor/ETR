# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Mainly copy-paste from https://github.com/facebookresearch/detr/blob/master/models/transformer.py
"""
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                d_model, 
                nhead, 
                dim_feedforward, 
                mha_dropout, 
                ffn_dropout, 
                activation, 
                normalize_before):
        super().__init__()

        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=mha_dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(ffn_dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(ffn_dropout)
        self.dropout2 = nn.Dropout(ffn_dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward_post(self,
                     src,
                     src_key_padding_mask,
                     tgt,
                     tgt_key_padding_mask,
                     ):
        # if self-attention the q,k,v is the same, either all src or all target
        q, k, v = src, tgt, tgt

        # MHA
        src2 = self.mha(query=q, key=k, value=v,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, 
                    src,
                    src_key_padding_mask,
                    tgt,
                    tgt_key_padding_mask,
                   ):
        src2 = self.norm1(src)
        q, k, v = src2, src2, src2

        # MHA 
        src2 = self.mha(query=q, key=k, value=v,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)

        # FFN
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, 
                src,
                src_key_padding_mask,
                tgt,
                tgt_key_padding_mask,
                ):

        if self.normalize_before:
            return self.forward_pre(src, src_key_padding_mask,
                                    tgt, tgt_key_padding_mask)
        else:
            return self.forward_post(src, src_key_padding_mask,
                                     tgt, tgt_key_padding_mask)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "elu":
        return F.elu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
