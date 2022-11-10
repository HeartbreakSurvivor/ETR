import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.modules import dropout
from torch.nn.modules.linear import Identity

class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            # nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.ffn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0., eps=1e-6):
        super().__init__()
        self.eps = eps

    def elu_feature_map(self, x):
        return torch.nn.functional.elu(x) + 1

    def forward(self, queries, keys, values):

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # print('qkv.shape: ', len(qkv), qkv[0].shape)

        # 将通过线性层后的矩阵，拆分成q k v 三个矩阵
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # print('q k v :', q.shape)

        Q = self.elu_feature_map(queries)
        K = self.elu_feature_map(keys)
        print('Q: ', Q.shape)
        print('K: ', K.shape)

        v_length = values.size(1)
        print('v_length: ', v_length)
        values = values / v_length  # prevent fp16 overflow
        print('values: ', values.shape)

        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        print('KV: ', KV.shape)
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        print('Z: ', Z.shape)

        out = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
        print("out: ", out.shape)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return out.contiguous()

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        # 每个head 以及对应head 的维度
        dim_head = dim // heads
        # inner_dim = dim_head * heads
        # print('dim_head, heads', dim_head, heads)

        # 如果heads为1，并且这个head的维度就跟最后输出的维度一样的时候，就不需要project_out
        project_out = not (heads == 1 and dim_head == dim)
        # print('project_out: ', project_out)

        self.heads = heads
        # 除以根号Dk,保持梯度稳定
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self._attn_map = None

        # 通过线性层映射成3个相同的矩阵
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim, dim),
        #     nn.Dropout(dropout)
        # ) if project_out else nn.Identity()

    def forward(self, x):
        # 根据输入x得到 qkv三个矩阵
        # 沿着最后一个通道，将to_qkv的结果分成3个tensor
        # print('Transformer input x: ', x.shape)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # print('qkv.shape: ', len(qkv), qkv[0].shape)

        # 将通过线性层后的矩阵，拆分成q k v 三个矩阵
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # print('q k v :', q.shape)
        # 计算 Q*K^T 再除以 根号dk
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # print('dots shape:', dots.shape)

        # 再经过softmax层得到概率分布 
        self._attn_map = self.attend(dots)
        # print('self._attn_map shape: ', self._attn_map.shape)

        # 再乘以v
        out = torch.matmul(self._attn_map, v)
        # print('before out: ', out.shape)
        
        # 再将多头得到的记过拼接起来
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print('after rearrange out: ', out.shape)
        # print('-------------- Quit Transformer Block\n ')
        # return self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, ffn_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreLayerNorm(dim, Attention(dim, heads = heads, dropout = dropout)),
                PreLayerNorm(dim, FeedForward(dim, ffn_dim, dropout = dropout))
            ]))

        self._attn_map = None

    def forward(self, x):
        for idx, (attn, ffn) in enumerate(self.layers):
            # 一个Transformer block包括一个 SA计算单元 + 一个FNN单元
            # 每个block之前需要先进行LayerNorm，每个block都通过残差连接
            # print('-------------- Enter {} Transformer Block'.format(idx))
            x = attn(x) + x
            x = ffn(x) + x
            # print('-------------- Quit {} Transformer Block'.format(idx))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        image_h, image_w = config['image_size']
        patch_h, patch_w = config['patch_size']
        assert image_h % patch_h == 0 and image_w % patch_w ==0, "the input image size must be divisible by the patch size"

        config_patch_dim = config['patch_dim']
        self.out_dim = config_patch_dim
        patch_dim = config['channels'] * patch_h * patch_w
        self.num_patches = (image_h // patch_h) * (image_w // patch_w)
        
        print('patch_dim, hidden_dim ', patch_dim, config_patch_dim)

        self.pool = config['pool']
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            # nn.Conv2d(patch_dim, hidden_dim, kernel_size=patch_h, stride=patch_h), # can be replace with conv operation
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_h, p2 = patch_w),
            nn.Linear(patch_dim, config_patch_dim),
        )
        # if config_patch_dim != patch_dim: # use the config patch dim
        #     self.out_dim = config_patch_dim
        #     self.to_config_patch_dim = nn.Linear(patch_dim, config_patch_dim)

        # print('self.outdim:', self.out_dim)

        # dropout rate
        self.dropout = nn.Dropout(config['emb_dropout'])
        # position encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, config_patch_dim))
        # class token
        # self.cls_token = nn.Parameter(torch.randn(1, 1, config_patch_dim))
        # transformer block
        self.transformer = Transformer(config_patch_dim, config['depth'], config['heads'], config['ffn_dim'], config['ffn_dropout'])

        # for classification task
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(config_patch_dim),
        #     nn.Linear(config_patch_dim, 81313)
        # )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            # print('p', p, p.dim())
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # print('..................--------------------------------------------------------------------------------------------------')
        # for m in self.modules():
        #     print('dddd', m)
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('input x: ', x.shape)
        b, _, _, _ = x.shape

        # 1. convert image to patch embedding...
        x = self.to_patch_embedding(x)
        # print('x1: ', x.shape)

        # 将cls_token 重复batch次，b是batch的数量，每次forward的batch size可能不同 
        # 2. concat class token with img
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)

        # 3. add position embedding...
        # print('position encoding shape: ', self.pos_embedding.shape)
        x += self.pos_embedding
        # print('x2: ', x.shape)

        x = self.dropout(x)

        # 3. through transformer block...
        x = self.transformer(x)
        # print('after transformer: ', x.shape)

        # for classification task
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # x = x.mean(dim=-1)
        # print('after pool: ', x.shape)

        # 只使用cls_token的输出来经过全连接层做分类
        # return self.mlp_head(x)
        return x

if __name__ == '__main__':
    encoder = TransformerEncoder(
        image_size = (30, 40),
        channels = 256,
        patch_size = (5, 5), 
        hidden_dim = 2048, 
        depth = 3,
        heads = 8,
        ffn_dim = 2048, 
        dropout = 0.1,
        emb_dropout = 0.1
    )

    x = torch.randn(1, 256, 30, 40)
    preds = encoder(x)
    print('out:', preds.shape)
