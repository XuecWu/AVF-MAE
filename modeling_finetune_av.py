\
\
\
\


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from functools import partial
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange, repeat
from modeling_attention_av import VisionTransformerEncoderForFusion_new_no_Mask_IR

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1        = nn.Linear(in_features, hidden_features)
        self.act        = act_layer()
        self.fc2        = nn.Linear(hidden_features, out_features)

        self.drop       = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)


        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim       = dim // num_heads

        if attn_head_dim is not None:
            head_dim   = attn_head_dim

        all_head_dim   = head_dim * self.num_heads
        self.scale     = qk_scale or head_dim ** -0.5

        self.qkv       = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj      = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, mask=None):
        B, N, C  = x.shape
        qkv_bias = None

        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))


        qkv     = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv     = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q       = q * self.scale
        attn    = (q @ k.transpose(-2, -1))


        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn     = self.attn_drop(attn)

        x        = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x        = self.proj(x)
        x        = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()

        self.norm1     = norm_layer(dim)
        self.attn      = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp       = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, mask=None):


        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


"""
adapted from https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
"""


class GeneralAttention(nn.Module):
    def __init__(self, dim, context_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim       = dim // num_heads

        if attn_head_dim is not None:
            head_dim   = attn_head_dim

        all_head_dim   = head_dim * self.num_heads

        self.scale     = qk_scale or head_dim ** -0.5

        self.q  = nn.Linear(dim, all_head_dim, bias=False)
        self.kv = nn.Linear(dim if context_dim is None else context_dim, all_head_dim * 2, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj      = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        B, T1, C        = x.shape
        q_bias, kv_bias = self.q_bias, None

        if self.q_bias is not None:
            kv_bias     = torch.cat((torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))


        q        = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q        = q.reshape(B, T1, self.num_heads, -1).transpose(1, 2)

        kv       = F.linear(input=x if context is None else context, weight=self.kv.weight, bias=kv_bias)

        _, T2, _ = kv.shape
        kv       = kv.reshape(B, T2, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v     = kv[0], kv[1]


        q        = q * self.scale
        attn     = (q @ k.transpose(-2, -1))

        attn     = attn.softmax(dim=-1)
        attn     = self.attn_drop(attn)

        x        = (attn @ v).transpose(1, 2).reshape(B, T1, -1)


        x        = self.proj(x)
        x        = self.proj_drop(x)

        return x


class LGBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None,

                 first_attn_type='self', third_attn_type='cross',
                 attn_param_sharing_first_third=False, attn_param_sharing_all=False,
                 no_second=False, no_third=False,
                 ):

        super().__init__()

        assert first_attn_type in ['self', 'cross'], f"Error: invalid attention type '{first_attn_type}', expected 'self' or 'cross'!"
        assert third_attn_type in ['self', 'cross'], f"Error: invalid attention type '{third_attn_type}', expected 'self' or 'cross'!"

        self.first_attn_type                = first_attn_type
        self.third_attn_type                = third_attn_type

        self.attn_param_sharing_first_third = attn_param_sharing_first_third
        self.attn_param_sharing_all         = attn_param_sharing_all


        self.first_attn_norm0     = norm_layer(dim)

        if self.first_attn_type  == 'cross':
            self.first_attn_norm1 = norm_layer(dim)


        self.first_attn           = GeneralAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)


        self.no_second             = no_second

        if not no_second:
            self.second_attn_norm0 = norm_layer(dim)


            if attn_param_sharing_all:
                self.second_attn   = self.first_attn
            else:
                self.second_attn   = GeneralAttention(
                    dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)


        self.no_third = no_third

        if not no_third:
            self.third_attn_norm0     = norm_layer(dim)

            if self.third_attn_type  == 'cross':
                self.third_attn_norm1 = norm_layer(dim)


            if attn_param_sharing_first_third or attn_param_sharing_all:
                self.third_attn = self.first_attn
            else:
                self.third_attn = GeneralAttention(
                    dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)


        if not no_third:

            self.fourth_attn_norm0   = norm_layer(dim)
            self.fourth_attn_norm1   = norm_layer(dim)

            self.fourth_attn         = GeneralAttention(
                dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2     = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp       = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_0 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_0, self.gamma_1, self.gamma_2 = None, None, None


    def forward(self, x, b):
        """
        :param x: (B*N, S, C),
            B: batch size
            N: number of local regions
            S: 1 + region size, 1: attached messenger token for each local region
            C: feature dim
        param b: batch size
        :return: (B*N, S, C)
        """
        bn = x.shape[0]
        n  = bn // b


        if self.gamma_1 is None:


            if self.first_attn_type == 'self':
                x = x + self.drop_path(self.first_attn(self.first_attn_norm0(x)))

            else:
                x[:,:1] = x[:,:1] + self.drop_path(
                    self.first_attn(
                        self.first_attn_norm0(x[:,:1]),
                        context=self.first_attn_norm1(x[:,1:])
                    )
                )


            if not self.no_second:


                messenger_tokens = messenger_tokens + self.drop_path(self.second_attn(self.second_attn_norm0(messenger_tokens)))

                x[:,0]           = rearrange(messenger_tokens, 'b n c -> (b n) c')

            else:
                messenger_tokens = rearrange(x[:,0], '(b n) c -> b n c', b=b)


            if not self.no_third:


                if self.third_attn_type == 'self':
                    x = x + self.drop_path(self.third_attn(self.third_attn_norm0(x)))


                else:


                    local_tokens = local_tokens + self.drop_path(
                        self.third_attn(
                            self.third_attn_norm0(local_tokens),
                            context=self.third_attn_norm1(messenger_tokens)
                        )
                    )

                    x[:,1:] = rearrange(local_tokens, 'b (n s) c -> (b n) s c', n=n)


            if self.fourth_attn:
                messenger_tokens_fourth      = messenger_tokens
                local_tokens_fourth          = local_tokens

                messenger_tokens_fourth      = messenger_tokens_fourth + self.drop_path(
                    self.fourth_attn(
                        self.fourth_attn_norm0(messenger_tokens_fourth),
                        context=self.fourth_attn_norm1(local_tokens_fourth)
                    )
                )


                x[:,0]           = rearrange(messenger_tokens_fourth, 'b n c -> (b n) c', b=b)


            x = x + self.drop_path(self.mlp(self.norm2(x)))

        else:
            raise NotImplementedError

        return x


class CSBlock(nn.Module):
    def __init__(self, dim, context_dim, num_heads, num_cross_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None, cross_attn_head_dim=None):
        super().__init__()


        self.cross_attn  = GeneralAttention(dim=dim, context_dim=context_dim, num_heads=num_cross_heads, qkv_bias=qkv_bias,
                                           qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=cross_attn_head_dim)

        self.cross_norm1 = norm_layer(dim)
        self.cross_norm2 = norm_layer(context_dim)


        self.norm1 = norm_layer(dim)
        self.attn  = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp       = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


        if init_values > 0:
            self.gamma_0 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_0, self.gamma_1, self.gamma_2 = None, None, None

    def forward(self, x, context):


        if self.gamma_1 is None:
            x = x + self.drop_path(self.cross_attn(self.cross_norm1(x), context=self.cross_norm2(context)))
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        else:
            x = x + self.drop_path(self.gamma_0 * self.cross_attn(self.cross_norm1(x), context=self.cross_norm2(context)))
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size          = to_2tuple(img_size)
        print(f'The real input size of video is {img_size}')

        patch_size        = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches       = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)

        self.img_size     = img_size
        self.patch_size   = patch_size
        self.num_patches  = num_patches


        self.temporal_seq_len    = num_frames // self.tubelet_size
        self.spatial_num_patches = num_patches // self.temporal_seq_len
        self.input_token_size    = (num_frames // self.tubelet_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])


        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]),
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape


        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."


        x             = self.proj(x).flatten(2).transpose(1, 2)

        return x


class PatchEmbed2D(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size         = to_2tuple(img_size)
        patch_size       = to_2tuple(patch_size)
        stride           = to_2tuple(stride)

        self.img_size    = img_size
        self.patch_size  = patch_size


        self.proj        = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)


        _, _, h, w       = self.get_output_shape(img_size)
        self.patch_hw    = (h, w)
        self.num_patches = h * w

        self.input_token_size = (h, w)


    def get_output_shape(self, img_size):

        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):


        B, C, H, W = x.shape
        x          = self.proj(x).flatten(2).transpose(1, 2)
        return x


def get_sinusoid_encoding_table(n_position, d_hid):
    '''
    Sinusoid position encoding table
    '''


    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]


    sinusoid_table          = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 keep_temporal_dim=False,
                 head_activation_func=None,
                 attn_type='joint',

                 lg_region_size=(2, 5, 10), lg_first_attn_type='self', lg_third_attn_type='cross',
                 lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False,
                 lg_classify_token_type='org', lg_no_second=False, lg_no_third=False,
                 ):
        super().__init__()

        self.num_classes  = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.tubelet_size = tubelet_size

        self.patch_embed  = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches       = self.patch_embed.num_patches


        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:


            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop      = nn.Dropout(p=drop_rate)
        dpr                = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]


        self.attn_type     = attn_type

        if attn_type      == 'joint':
            print(f'using the attention type of {attn_type}')
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values
                )
                for i in range(depth)])

        elif attn_type == 'local_global':
            print(f"==> Note for Video: We use 'local_global' for compute reduction (lg_region_size={lg_region_size},"
                  f"lg_first_attn_type={lg_first_attn_type}, lg_third_attn_type={lg_third_attn_type},"
                  f"lg_attn_param_sharing_first_third={lg_attn_param_sharing_first_third},"
                  f"lg_attn_param_sharing_all={lg_attn_param_sharing_all},"
                  f"lg_classify_token_type={lg_classify_token_type},"
                  f"lg_no_second={lg_no_second}, lg_no_third={lg_no_third})")

            self.blocks = nn.ModuleList([
                LGBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values,
                    first_attn_type=lg_first_attn_type, third_attn_type=lg_third_attn_type,
                    attn_param_sharing_first_third=lg_attn_param_sharing_first_third,
                    attn_param_sharing_all=lg_attn_param_sharing_all,
                    no_second=lg_no_second, no_third=lg_no_third,
                )
                for i in range(depth)])


            self.lg_region_size         = lg_region_size
            print(f"The t, h, and w of lg_region_size are {self.lg_region_size[0], self.lg_region_size[1], self.lg_region_size[2]}")

            self.lg_num_region_size     = list(i//j for i,j in zip(self.patch_embed.input_token_size, lg_region_size))
            num_regions                 = self.lg_num_region_size[0] * self.lg_num_region_size[1] * self.lg_num_region_size[2]
            print(f"==> The number of local regions: {num_regions} (size={self.lg_num_region_size})")


            self.lg_region_tokens       = nn.Parameter(torch.zeros(num_regions, embed_dim))
            trunc_normal_(self.lg_region_tokens, std=.02)


            self.lg_classify_token_type = lg_classify_token_type
            assert lg_classify_token_type in ['org', 'region', 'all'], f"Error: wrong 'lg_classify_token_type' in local_global attention ('{lg_classify_token_type}'), expected 'org'/'region'/'all'!"

        else:
            raise NotImplementedError

        self.norm    = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head    = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()


        self.keep_temporal_dim = keep_temporal_dim


        if head_activation_func is not None:
            if head_activation_func      == 'sigmoid':
                self.head_activation_func = nn.Sigmoid()
            elif head_activation_func    == 'relu':
                self.head_activation_func = nn.ReLU()
            elif head_activation_func    == 'tanh':
                self.head_activation_func = nn.Tanh()
            else:
                raise NotImplementedError
        else:
            self.head_activation_func     = nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        if num_classes > 0:
            trunc_normal_(self.head.weight, std=.02)

        self.apply(self._init_weights)

        if num_classes > 0:
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'part_tokens', 'lg_region_tokens'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head        = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):


        x       = self.patch_embed(x)
        B, _, _ = x.size()


        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()

        x = self.pos_drop(x)


        if self.attn_type == 'local_global':


            nt, t = self.lg_num_region_size[0], self.lg_region_size[0]
            nh, h = self.lg_num_region_size[1], self.lg_region_size[1]
            nw, w = self.lg_num_region_size[2], self.lg_region_size[2]

            b     = x.size(0)
            x     = rearrange(x, 'b (nt t nh h nw w) c -> b (nt nh nw) (t h w) c', nt=nt, nh=nh, nw=nw, t=t, h=h, w=w)


            region_tokens = repeat(self.lg_region_tokens, 'n c -> b n 1 c', b=b)

            x             = torch.cat([region_tokens, x], dim=2)

            x             = rearrange(x, 'b n s c -> (b n) s c')


            intermediate_features = []
            for blk in self.blocks:

                x     = blk(x, b)
                x_new = rearrange(x, '(b n) s c -> b n s c', b=b)


                if self.lg_classify_token_type   == 'region':
                    x_new = x_new[:,:,0]

                elif self.lg_classify_token_type == 'org':
                    x_new = rearrange(x_new[:,:,1:], 'b n s c -> b (n s) c')

                else:
                    x_new = rearrange(x_new, 'b n s c -> b (n s) c')

                intermediate_features.append(x_new)

        else:


            intermediate_features = []
            for blk in self.blocks:
                x = blk(x)
                intermediate_features.append(x)


        x = torch.stack(intermediate_features, dim=2)


        x = self.norm(x)

        if self.fc_norm is not None:


            if self.keep_temporal_dim:
                x = rearrange(x, 'b (t hw) c -> b c t hw', t=self.patch_embed.temporal_seq_len, hw=self.patch_embed.spatial_num_patches)


                x = x.mean(-1)


                x = torch.nn.functional.interpolate(x, scale_factor=self.patch_embed.tubelet_size, mode='linear')

                x = rearrange(x, 'b c t -> b t c')

                return self.fc_norm(x)

            else:
                return self.fc_norm(x.mean(1))
        else:
            return x

    def forward(self, x):

        x = self.forward_features(x)


        x = self.head(x)


        x = self.head_activation_func(x)


        if self.keep_temporal_dim:
            x = x.view(x.size(0), -1)

        return x


class VisionTransformerEncoder2D(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_checkpoint=False,
                 use_learnable_pos_emb=False, init_scale=0., use_mean_pooling=True,

                 attn_type='joint',
                 lg_region_size_audio=(4, 4), lg_first_attn_type='self', lg_third_attn_type='cross',
                 lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False,
                 lg_classify_token_type='org', lg_no_second=False, lg_no_third=False,
                 ):
        super().__init__()

        self.num_classes    = num_classes
        self.num_features   = self.embed_dim = embed_dim

        self.patch_embed    = PatchEmbed2D(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=patch_size)
        num_patches         = self.patch_embed.num_patches

        self.use_checkpoint = use_checkpoint


        if use_learnable_pos_emb:
            self.pos_embed  = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:

            self.pos_embed  = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr           = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]


        self.attn_type     = attn_type


        if attn_type      == 'joint':
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values
                )
                for i in range(depth)])

        elif attn_type == 'local_global':
            print(f"==> Note for AUDIO: We use 'local_global' for compute reduction (lg_region_size_audio={lg_region_size_audio},"
                  f"lg_first_attn_type={lg_first_attn_type}, lg_third_attn_type={lg_third_attn_type},"
                  f"lg_attn_param_sharing_first_third={lg_attn_param_sharing_first_third},"
                  f"lg_attn_param_sharing_all={lg_attn_param_sharing_all},"
                  f"lg_classify_token_type={lg_classify_token_type},"
                  f"lg_no_second={lg_no_second}, lg_no_third={lg_no_third})")

            self.blocks = nn.ModuleList([
                LGBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values,
                    first_attn_type=lg_first_attn_type, third_attn_type=lg_third_attn_type,
                    attn_param_sharing_first_third=lg_attn_param_sharing_first_third,
                    attn_param_sharing_all=lg_attn_param_sharing_all,
                    no_second=lg_no_second, no_third=lg_no_third,
                )
                for i in range(depth)])


            self.lg_region_size         = lg_region_size_audio
            print(f"The h and w of lg_region_size for AUDIO are {self.lg_region_size[0], self.lg_region_size[1]}")

            self.lg_num_region_size     = list(i//j for i,j in zip(self.patch_embed.input_token_size, lg_region_size_audio))
            num_regions                 = self.lg_num_region_size[0] * self.lg_num_region_size[1]
            print(f"==> The number of local regions for Auido is : {num_regions} (size={self.lg_num_region_size})")


            self.lg_region_tokens       = nn.Parameter(torch.zeros(num_regions, embed_dim))
            trunc_normal_(self.lg_region_tokens, std=.02)


            self.lg_classify_token_type = lg_classify_token_type
            assert lg_classify_token_type in ['org', 'region', 'all'], f"Error: wrong 'lg_classify_token_type' in local_global attention ('{lg_classify_token_type}'), expected 'org'/'region'/'all'!"

        else:
            raise NotImplementedError


        self.norm    = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

        self.head    = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        if num_classes > 0:
            trunc_normal_(self.head.weight, std=.02)

        self.apply(self._init_weights)

        if num_classes > 0:
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'part_tokens', 'lg_region_tokens'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head        = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):


        x       = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x   = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()

        x       = self.pos_drop(x)


        if self.attn_type == 'local_global':


            nh, h = self.lg_num_region_size[0], self.lg_region_size[0]
            nw, w = self.lg_num_region_size[1], self.lg_region_size[1]

            b     = x.size(0)
            x     = rearrange(x, 'b (nh h nw w) c -> b (nh nw) (h w) c', nh=nh, nw=nw, h=h, w=w)


            region_tokens = repeat(self.lg_region_tokens, 'n c -> b n 1 c', b=b)

            x             = torch.cat([region_tokens, x], dim=2)

            x             = rearrange(x, 'b n s c -> (b n) s c')


            intermediate_features = []
            for blk in self.blocks:


                x     = blk(x, b)
                x_new = rearrange(x, '(b n) s c -> b n s c', b=b)


                if self.lg_classify_token_type   == 'region':
                    x_new = x_new[:,:,0]

                elif self.lg_classify_token_type == 'org':
                    x_new = rearrange(x_new[:,:,1:], 'b n s c -> b (n s) c')

                else:
                    x_new = rearrange(x_new, 'b n s c -> b (n s) c')

                intermediate_features.append(x_new)


        else:


            intermediate_features = []
            for blk in self.blocks:
                x = blk(x)
                intermediate_features.append(x)


        x = torch.stack(intermediate_features, dim=2)


        x = self.norm(x)

        if self.fc_norm is not None:

            return self.fc_norm(x.mean(1))
        else:
            return x


    def forward(self, x):
        x = self.forward_features(x)


        x = self.head(x)

        return x


class VisionTransformerEncoderForFusion(nn.Module):
    def __init__(self, embed_dim=768, embed_dim_audio=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,):
        super().__init__()

        dpr         = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]


        self.blocks = nn.ModuleList([
            CSBlock(dim=embed_dim, context_dim=embed_dim_audio, num_heads=num_heads,
                    num_cross_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)


        self.blocks_audio = nn.ModuleList([
            CSBlock(dim=embed_dim_audio, context_dim=embed_dim, num_heads=num_heads,
                    num_cross_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
            for i in range(depth)])
        self.norm_audio = norm_layer(embed_dim_audio)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head        = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward(self, x, x_audio):


        for blk, blk_audio in zip(self.blocks, self.blocks_audio):
            x, x_audio = blk(x, context=x_audio), blk_audio(x_audio, context=x)


        x       = self.norm(x)
        x_audio = self.norm_audio(x_audio)

        return x, x_audio


class AudioVisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 keep_temporal_dim=False,
                 attn_type='joint',
                 img_size_audio=(256, 128),
                 patch_size_audio=16,
                 in_chans_audio=1,
                 embed_dim_audio=768,
                 depth_audio=12,
                 num_heads_audio=12,
                 fusion_depth=2,
                 fusion_num_heads=12,
                 use_mean_pooling=False,
                 head_activation_func=None,

                 lg_region_size=(2, 5, 10),
                 lg_region_size_audio=(4, 4),
                 lg_first_attn_type='self', lg_third_attn_type='cross',
                 lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False, lg_classify_token_type='region', lg_no_second=False, lg_no_third=False,
                 ):
        super().__init__()


        self.encoder_depth = depth


        if depth > 0 :
            self.encoder = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_classes=0,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                init_values=init_values,
                init_scale=init_scale,
                all_frames=all_frames,
                tubelet_size=tubelet_size,
                use_learnable_pos_emb=use_learnable_pos_emb,
                use_mean_pooling=False,
                attn_type=attn_type,

                lg_region_size=lg_region_size,
                lg_first_attn_type=lg_first_attn_type, lg_third_attn_type=lg_third_attn_type,
                lg_attn_param_sharing_first_third=lg_attn_param_sharing_first_third, lg_attn_param_sharing_all=lg_attn_param_sharing_all,
                lg_classify_token_type=lg_classify_token_type, lg_no_second=lg_no_second, lg_no_third=lg_no_third,
            )

        else:
            self.encoder = None
            print(f"==> Warning: video-specific encoder is not used!!!")


        self.encoder_depth_audio = depth_audio


        if depth_audio > 0:
            self.encoder_audio = VisionTransformerEncoder2D(
                img_size=img_size_audio,
                patch_size=patch_size_audio,
                in_chans=in_chans_audio,
                num_classes=0,
                embed_dim=embed_dim_audio,
                depth=depth_audio,
                num_heads=num_heads_audio,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                init_values=init_values,
                use_learnable_pos_emb=use_learnable_pos_emb,
                use_mean_pooling=False,

                attn_type=attn_type,
                lg_region_size_audio=lg_region_size_audio,
                lg_first_attn_type=lg_first_attn_type, lg_third_attn_type=lg_third_attn_type,
                lg_attn_param_sharing_first_third=lg_attn_param_sharing_first_third, lg_attn_param_sharing_all=lg_attn_param_sharing_all,
                lg_classify_token_type=lg_classify_token_type, lg_no_second=lg_no_second, lg_no_third=lg_no_third,
            )

        else:
            self.encoder_audio = None
            print(f"==> Warning: audio-specific encoder is not used!!!")

        print(f'The embedding dims of video and audio encoders are {embed_dim} and {embed_dim_audio}, respectively.')
        print(f'The number of attention heads in video and audio encoders are {num_heads} and {num_heads_audio}, respectively.')
        print(f'The depths of video and audio encoders are {depth} and {depth_audio}, respectively.')


        if fusion_depth > 0 and self.encoder is not None and self.encoder_audio is not None:
            self.encoder_fusion = VisionTransformerEncoderForFusion_new_no_Mask_IR(
                embed_dim=embed_dim,
                embed_dim_audio=embed_dim_audio,
                depth=fusion_depth,
                num_heads=fusion_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=0,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer,
                init_values=init_values,
            )
            self.encoder_depth_fusion = fusion_depth
            print(f"==> Note for Audio-Viusal FUSION: the depth of fusion encoder is {fusion_depth}, the attention heads of fusion is {fusion_num_heads}, and Iterative Refinement is deployed!!!")
        else:
            self.encoder_fusion       = None
            self.encoder_depth_fusion = 0
            print(f"==> Warning: fusion encoder is not used!!!")


        fc_num       = embed_dim + embed_dim_audio
        print(f'The final dims (fc_num) is {fc_num}')


        self.fc_norm = norm_layer(fc_num)
        self.head    = nn.Linear(fc_num, num_classes) if num_classes > 0 else nn.Identity()
        print(f'The final dimension of head (classes) is {num_classes}')


        if head_activation_func is not None:
            if head_activation_func == 'sigmoid':
                self.head_activation_func = nn.Sigmoid()
            elif head_activation_func == 'relu':
                self.head_activation_func = nn.ReLU()
            elif head_activation_func == 'tanh':
                self.head_activation_func = nn.Tanh()
            else:
                raise NotImplementedError
        else:
            self.head_activation_func = nn.Identity()


        self.layer_weights_video = nn.Parameter(torch.ones(self.encoder_depth) / self.encoder_depth)
        self.layer_weights_audio = nn.Parameter(torch.ones(self.encoder_depth_audio) / self.encoder_depth_audio)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def get_num_layers(self):

        return max(self.encoder_depth, self.encoder_depth_audio) +\
               self.encoder_depth_fusion

    def get_num_modality_specific_layers(self):

        return max(self.encoder_depth, self.encoder_depth_audio)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'part_tokens', 'lg_region_tokens',
                'encoder.pos_embed', 'encoder_audio.pos_embed', 'encoder_fusion.pos_embed',
                'encoder.cls_token', 'encoder_audio.cls_token', 'encoder_fusion.cls_token',}


    def forward_features(self, x, x_audio):


        x       = self.encoder(x)


        x_audio = self.encoder_audio(x_audio)


        layer_weights_video = torch.nn.functional.softmax(self.layer_weights_video, dim=0)
        layer_weights_audio = torch.nn.functional.softmax(self.layer_weights_audio, dim=0)

        video_feature       = torch.einsum('bnlc,l->bnc', x, layer_weights_video)
        audio_feature       = torch.einsum('bnlc,l->bnc', x_audio, layer_weights_audio)

        overall_feat        = torch.cat([video_feature, audio_feature], dim=-1)


        x                    = x[:,:,-1]
        x_audio              = x_audio[:,:,-1]


        x, x_audio, _  = self.encoder_fusion(x, x_audio, overall_feat)


        fusion_video_feature   = x.mean(dim=1)
        fusion_audio_feature   = x_audio.mean(dim=1)


        final_feature          = torch.cat([fusion_video_feature, fusion_audio_feature], dim=-1)


        return self.fc_norm(final_feature)


    def forward(self, x, x_audio, save_feature=False):


        x = self.forward_features(x, x_audio)


        if save_feature:
            feature = x

        x = self.head(x)


        x = self.head_activation_func(x)


        if save_feature:
            return x, feature
        else:
            return x


@register_model
def avit_dim512_patch16_160_a256(pretrained=False, **kwargs):
    embed_dim  = 512
    num_heads  = 8
    patch_size = 16

    model = AudioVisionTransformer(

        img_size=160,
        patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,

        img_size_audio=(256, 128),
        patch_size_audio=patch_size, embed_dim_audio=embed_dim, num_heads_audio=num_heads,

        fusion_num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()

    return model


@register_model
def avit_dim640_patch16_160_a256(pretrained=False, **kwargs):
    embed_dim  = 640
    num_heads  = 10
    patch_size = 16

    model = AudioVisionTransformer(

        img_size=160,
        patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,

        img_size_audio=(256, 128),
        patch_size_audio=patch_size, embed_dim_audio=embed_dim, num_heads_audio=num_heads,

        fusion_num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()

    return model


@register_model
def avit_dim768_patch16_160_a256(pretrained=False, **kwargs):
    embed_dim  = 768
    num_heads  = 12
    patch_size = 16

    model = AudioVisionTransformer(

        img_size=160,
        patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,

        img_size_audio=(256, 128),
        patch_size_audio=patch_size, embed_dim_audio=embed_dim, num_heads_audio=num_heads,

        fusion_num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()

    return model
