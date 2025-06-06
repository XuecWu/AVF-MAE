import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from functools import partial
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange, repeat
from modeling_attention_av import VisionTransformerEncoderForFusion_new_no_Mask_IR # [用于整体微调模型]

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


#----------------------------#
# 这个是在后面的Block中被使用
#----------------------------#
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

#----------------------#
# MLP的定义
# 用于构建Block
# 存在单独的dropout定义
#----------------------#
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

        # x = self.drop(x)
        # commit this for the orignal BERT implement 

        x = self.fc2(x)
        x = self.drop(x)

        return x

#------------------------------#
# 通用注意力模块的构建
#------------------------------#
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim       = dim // num_heads

        if attn_head_dim is not None:
            head_dim   = attn_head_dim

        all_head_dim   = head_dim * self.num_heads
        self.scale     = qk_scale or head_dim ** -0.5 # 公式中定义的缩放因子

        self.qkv       = nn.Linear(dim, all_head_dim * 3, bias=False) # 存在weight, 但是不存在bias

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj      = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    #----------------------------------#
    # Attention的前向传播过程
    # 这里的mask可以在具体应用时给传进去
    #----------------------------------#
    def forward(self, x, mask=None): # NOTE: 注意这里要mask
        B, N, C  = x.shape
        qkv_bias = None

        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            
        qkv     = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv     = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q       = q * self.scale # 完成公式中Q与缩放因子的作用
        attn    = (q @ k.transpose(-2, -1)) # 完成公式表达中softmax内部的操作
        

        # me: support window mask
        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1) # 过softmax得到输出
        else:
            attn = attn.softmax(dim=-1) # 过softmax得到输出

        attn     = self.attn_drop(attn) # 完成Att计算后, 过下dropout

        x        = (attn @ v).transpose(1, 2).reshape(B, N, -1) # 完成softmax部分和V部分的共同作用 

        x        = self.proj(x) # 通道数调整
        x        = self.proj_drop(x)

        return x


#------------------------------------------#
# 基于Attention与MLP等构建Block
# Block相当于为VideoMAE构建的原生注意力块
# NOTE: Pure Attention
#------------------------------------------#
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()

        self.norm1     = norm_layer(dim)
        self.attn      = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
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

        #-------------------------------------------------------------#
        # 两种情况的主要区别在于是否在attn和mlp的基础上乘以gamma系数
        # 其余的流程是保持一致的
        #-------------------------------------------------------------#
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask)) # mask是传给Attention这个类中的前向传播函数
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


"""
adapted from https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
"""
#------------------------------#
# support cross attention
# 主要用于视听跨模态注意力的定义
#------------------------------#
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

        #------------------------------#
        # 这里对context进行了调用
        # 如果非空, 则kv是context
        # 从而可以进行cross-attention
        #------------------------------#
        q        = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q        = q.reshape(B, T1, self.num_heads, -1).transpose(1, 2) # me: (B, H, T1, C//H)

        kv       = F.linear(input=x if context is None else context, weight=self.kv.weight, bias=kv_bias) # 这里实现对于context的使用, 如果是自注意力则context为None, 如果是跨模态注意力则context非None

        _, T2, _ = kv.shape
        kv       = kv.reshape(B, T2, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v     = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple), me： (B, H, T2, C//H)

        #---------------------------------------#
        # 在完成QKV的定义后开始进行attention的计算
        #---------------------------------------#
        q        = q * self.scale
        attn     = (q @ k.transpose(-2, -1)) # me: (B, H, T1, T2)

        attn     = attn.softmax(dim=-1)
        attn     = self.attn_drop(attn)

        x        = (attn @ v).transpose(1, 2).reshape(B, T1, -1) # (B, H, T1, C//H) -> (B, T1, H, C//H) -> (B, T1, C) # 完成attention的计算

        #----------------------------#
        # 通过映射层进行维度数的调整
        #----------------------------#
        x        = self.proj(x)
        x        = self.proj_drop(x)

        return x


#-------------------------------------------------------#
# local + global, 即为LGBlock的定义
# 1. 首先运用到模态编码器中去 (2024.6.26 19.17)
# 2. 跨模态融合器再说
# 3. 预训练模型中的decoder是不需要引入的, follow原始工作
#-------------------------------------------------------#
class LGBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None,
                 # new added # 6个参数
                 first_attn_type='self', third_attn_type='cross',
                 attn_param_sharing_first_third=False, attn_param_sharing_all=False,
                 no_second=False, no_third=False,
                 ):

        super().__init__()

        assert first_attn_type in ['self', 'cross'], f"Error: invalid attention type '{first_attn_type}', expected 'self' or 'cross'!"
        assert third_attn_type in ['self', 'cross'], f"Error: invalid attention type '{third_attn_type}', expected 'self' or 'cross'!"

        self.first_attn_type                = first_attn_type # self
        self.third_attn_type                = third_attn_type # cross

        self.attn_param_sharing_first_third = attn_param_sharing_first_third # 没有开启
        self.attn_param_sharing_all         = attn_param_sharing_all         # 没有开启

        #------------------------#
        # Attention layers
        #------------------------#

        ## perform local (intra-region) attention, update messenger tokens  # 区域内部
        ## (local->messenger) or (local<->local, local<->messenger)

        self.first_attn_norm0     = norm_layer(dim)

        if self.first_attn_type  == 'cross': # 我们为self 因此没有cross的作用
            self.first_attn_norm1 = norm_layer(dim)

        #--------------------------------------#
        # 定义出来用于区域内部建模的第一层注意力
        #--------------------------------------#
        self.first_attn           = GeneralAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)


        ## perform global (inter-region) attention on messenger tokens # 区域间
        ## (messenger<->messenger)

        self.no_second             = no_second

        if not no_second:
            self.second_attn_norm0 = norm_layer(dim)

            #----------------------------#
            # 定义第二层的注意力函数
            #----------------------------#
            if attn_param_sharing_all: # 如果三种参数均共享, 那么定义的一样即可, 否则就重新定义一个新的
                self.second_attn   = self.first_attn
            else:
                self.second_attn   = GeneralAttention( 
                    dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)


        ## perform local (intra-region) attention to inject global information into local tokens
        ## (messenger->local) or (local<->local, local<->messenger)
                
        self.no_third = no_third

        if not no_third:
            self.third_attn_norm0     = norm_layer(dim)

            if self.third_attn_type  == 'cross':
                self.third_attn_norm1 = norm_layer(dim)

            #------------------------------------#
            # 如果开启第一个与第三个的参数共享
            # 或者全部的参数共享
            # 那么第三层注意力与之前定义的就将一致
            # 否则就将重新定义一个
            #------------------------------------#
            if attn_param_sharing_first_third or attn_param_sharing_all:
                self.third_attn = self.first_attn
            else:
                self.third_attn = GeneralAttention(
                    dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        
        #--------------------------------------------#
        # NOTE:面向region tokens, 即为第四层注意力
        # 本质上为第三部分的反操作, 即Q与KV的取值相反
        # me: new added [2024.7.9 22.32]
        #--------------------------------------------#
        if not no_third:

            self.fourth_attn_norm0   = norm_layer(dim)
            self.fourth_attn_norm1   = norm_layer(dim)

            self.fourth_attn         = GeneralAttention(
                dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)


        # NOTE: 此时, 四层注意力和相关的norm layers都定义完毕, 接下来开始定义前馈层和其他补充层

        #------------#
        # FFN layer
        #------------#
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        self.norm2     = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp       = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_0 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_0, self.gamma_1, self.gamma_2 = None, None, None
    
    #-----------------------------------------------------------------#
    # 两个输入, 第一个为张量, 第二个为batch的大小
    # 我们新加入的注意力即为强化region token在多层LGBlock中的信息传递效力
    #-----------------------------------------------------------------#
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
        n  = bn // b # number of local regions, i.e., N
        
        #-------------------------------------------#
        # 根据这里的定义看下来, 这里的gamma都是空的
        #-------------------------------------------#
        if self.gamma_1 is None:
            

            if self.first_attn_type == 'self': # by default
                x = x + self.drop_path(self.first_attn(self.first_attn_norm0(x)))

            else: # 'cross' # NOTE: 如果为cross, 则为原始tokens与单独的message token进行交互
                x[:,:1] = x[:,:1] + self.drop_path(
                    self.first_attn(
                        self.first_attn_norm0(x[:,:1]), # (b*n, 1, c)
                        context=self.first_attn_norm1(x[:,1:]) # (b*n, s-1, c)
                    )
                )

            #--------------------------------------------------#
            # 2. perform global (inter-region) self-attention
            # 我们走if的流程
            #--------------------------------------------------#
            if not self.no_second: # NOTE: by default

                messenger_tokens = rearrange(x[:,0].clone(), '(b n) c -> b n c', b=b) # attn on 'n' dim # x[:,0] -> x[:,0].clone()  #https://github.com/sunlicai/MAE-DFER/issues/3  #2024.5.6 12.26
                messenger_tokens = messenger_tokens + self.drop_path(self.second_attn(self.second_attn_norm0(messenger_tokens))) # 常规的注意力操作流程, 只是输入存在区别

                x[:,0]           = rearrange(messenger_tokens, 'b n c -> (b n) c') # 将messenger_tokens重新插入原始张量

            else: # for usage in the third attn
                messenger_tokens = rearrange(x[:,0], '(b n) c -> b n c', b=b) # attn on 'n' dim 

            #-----------------------------------------#
            # 3. perform local-global interaction
            # 基于2中生成的x与messenger_tokens进行操作
            # 直接走if的流程即可
            #-----------------------------------------#
            if not self.no_third:
                #------------------#
                # self-attention
                #------------------#
                if self.third_attn_type == 'self':
                    x = x + self.drop_path(self.third_attn(self.third_attn_norm0(x)))

                #------------------#
                # cross-attention
                #------------------#
                else: # NOTE: by default

                    local_tokens = rearrange(x[:,1:].clone(), '(b n) s c -> b (n s) c', b=b) # NOTE: n merges into s (not b), (B, N*(S-1), D)  # x[:,1:] -> x[:,1:].clone() #https://github.com/sunlicai/MAE-DFER/issues/3 #2024.5.6 12.26
                    local_tokens = local_tokens + self.drop_path(
                        self.third_attn(
                            self.third_attn_norm0(local_tokens), # (b, n*(s-1), c)
                            context=self.third_attn_norm1(messenger_tokens) # (b, n*1, c)
                        )
                    )

                    x[:,1:] = rearrange(local_tokens, 'b (n s) c -> (b n) s c', n=n) # [B*N, S, C] # 对张量进行更新


            if self.fourth_attn: # NOTE: The shape of x = [B*N, S, C]
                messenger_tokens_fourth      = messenger_tokens
                local_tokens_fourth          = local_tokens

                messenger_tokens_fourth      = messenger_tokens_fourth + self.drop_path(
                    self.fourth_attn(
                        self.fourth_attn_norm0(messenger_tokens_fourth),    # (b, n*1, c)
                        context=self.fourth_attn_norm1(local_tokens_fourth) # (b, n*(s-1), c)
                    )
                )


                x[:,0]           = rearrange(messenger_tokens_fourth, 'b n c -> (b n) c', b=b) # 将messenger_tokens_fourth重新插入原始张量


            x = x + self.drop_path(self.mlp(self.norm2(x)))

        else:
            raise NotImplementedError
        
        return x 


class CSBlock(nn.Module):
    def __init__(self, dim, context_dim, num_heads, num_cross_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None, cross_attn_head_dim=None):
        super().__init__()


        self.cross_attn  = GeneralAttention(dim=dim, context_dim=context_dim, num_heads=num_cross_heads, qkv_bias=qkv_bias, 
                                           qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=cross_attn_head_dim) # 跨注意力
        
        self.cross_norm1 = norm_layer(dim) # Q related
        self.cross_norm2 = norm_layer(context_dim) # KV related


        self.norm1 = norm_layer(dim)
        self.attn  = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim) # 自注意力


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        #-------------------------------------#
        # 两者是一起使用的, 先是norm2, 再是MLP
        #-------------------------------------#
        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp       = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        #---------------------------#
        # 初始值大于0的情况下, 相当
        # 于给每一步都提取加上一个因子
        #---------------------------#
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

        else: # NOTE: 主要是多了gamma系数
            x = x + self.drop_path(self.gamma_0 * self.cross_attn(self.cross_norm1(x), context=self.cross_norm2(context)))
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


#------------------------------#
# 用于视频的patch embedding层
# Towards Video
#------------------------------#
class PatchEmbed(nn.Module):
    """ 
    Image to Patch Embedding for Videos.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size          = to_2tuple(img_size) # timm导入的方法, 相当于将一个元素扩展为了元组
        print(f'The real input size of video is {img_size}')

        patch_size        = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches       = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size) # 为 8*10*10=800

        self.img_size     = img_size
        self.patch_size   = patch_size
        self.num_patches  = num_patches

        # me: for more attention types
        self.temporal_seq_len    = num_frames // self.tubelet_size # 定义但实际未使用
        self.spatial_num_patches = num_patches // self.temporal_seq_len # 定义但实际未使用
        self.input_token_size    = (num_frames // self.tubelet_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1]) # (nt, nh, nw) = (8, 10, 10) 

        #---------------------------------#
        # 常见操作: 使用3D卷积进行shape转变
        #---------------------------------#
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape # 拿出shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        #---------------------------------------------#
        # 本质上还是通过3D卷积操作来进行patch embedding
        #---------------------------------------------#
        x             = self.proj(x).flatten(2).transpose(1, 2) # 进行shape的转变

        return x

class PatchEmbed2D(nn.Module):
    """ 
    Flexible Image to Patch Embedding for Audio
    We modify it to align with LGBlock
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size         = to_2tuple(img_size) # timm导入
        patch_size       = to_2tuple(patch_size)
        stride           = to_2tuple(stride)

        self.img_size    = img_size
        self.patch_size  = patch_size

        #--------------------------------#
        # 相当于此处为2D卷积, 视频为3D卷积
        #--------------------------------#
        self.proj        = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)  # with overlapped patches

        #---------------------------------#
        # 得到转换后的h与w, 并进行新的定义
        #---------------------------------#
        _, _, h, w       = self.get_output_shape(img_size)  # [n, emb_dim, h, w]
        self.patch_hw    = (h, w)
        self.num_patches = h * w

        self.input_token_size = (h, w) # 按照t h w的顺序, 我们省略掉t, 所以直接为h和w (h, w) = (16, 8)
    
    #-------------------------------------------------#
    # 对上面的映射层进行使用从而得到转变后的张量
    # 可以看出前两个维度是为了match 2D卷积的形式所添加的
    #-------------------------------------------------#
    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):
        #-------------------------------------------------#
        # 从此处的实现可以看出: 上述定义的方法并没有全部使用
        # 使用的只有proj函数 使用了2D卷积来进行shape转换
        #-------------------------------------------------#
        B, C, H, W = x.shape
        x          = self.proj(x).flatten(2).transpose(1, 2) # (B, N, C)
        return x


def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' 
    Sinusoid position encoding table 
    ''' 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table          = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 偶数
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 奇数

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 


class VisionTransformer(nn.Module):
    """ 
    For Video
    Vision Transformer with support for patch or hybrid CNN input stage
    """
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
                 keep_temporal_dim=False, # do not perform temporal pooling, has higher priority than 'use_mean_pooling'
                 head_activation_func=None, # activation function after head fc, mainly for the regression task
                 attn_type='joint',
                 # me: new added [2024.6.26 19.23]
                 lg_region_size=(2, 5, 10), lg_first_attn_type='self', lg_third_attn_type='cross',  # for local_global # 此为新添加的
                 lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False,
                 lg_classify_token_type='org', lg_no_second=False, lg_no_third=False,
                 ):
        super().__init__()

        self.num_classes  = num_classes # 后面我们使用的时候会传类别数进去
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tubelet_size = tubelet_size

        self.patch_embed  = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size) # patch embedding的定义 # 进行通道数调整
        num_patches       = self.patch_embed.num_patches

        #---------------------------#
        # 位置信息编码相关
        #---------------------------#
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:

            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim) 

        self.pos_drop      = nn.Dropout(p=drop_rate)
        dpr                = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule


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
            
            self.lg_region_size         = lg_region_size # (t, h, w) 我们在sh中指定的为(2, 5, 10)
            print(f"The t, h, and w of lg_region_size are {self.lg_region_size[0], self.lg_region_size[1], self.lg_region_size[2]}")

            self.lg_num_region_size     = list(i//j for i,j in zip(self.patch_embed.input_token_size, lg_region_size)) # (nt, nh, nw) 8 10 10 分别除以 2 5 10 得到结果为 4 2 1
            num_regions                 = self.lg_num_region_size[0] * self.lg_num_region_size[1] * self.lg_num_region_size[2] # nt * nh * nw # 相当于为8个regions
            print(f"==> The number of local regions: {num_regions} (size={self.lg_num_region_size})") # 实际输出与我们的验证结果一致 # 相当于为8个regions

            self.lg_region_tokens       = nn.Parameter(torch.zeros(num_regions, embed_dim)) # shape为 [8, 512]
            trunc_normal_(self.lg_region_tokens, std=.02)

            self.lg_classify_token_type = lg_classify_token_type # 默认为region
            assert lg_classify_token_type in ['org', 'region', 'all'], f"Error: wrong 'lg_classify_token_type' in local_global attention ('{lg_classify_token_type}'), expected 'org'/'region'/'all'!"

        else:
            raise NotImplementedError

        self.norm    = nn.Identity() if use_mean_pooling else norm_layer(embed_dim) # 存在
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None # 为空
        self.head    = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity() # 分类头 # 为空, 没有作用

        # me: add frame-level prediction support
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
        else: # default
            self.head_activation_func     = nn.Identity()

        if use_learnable_pos_emb: # no use
            trunc_normal_(self.pos_embed, std=.02)

        if num_classes > 0: # use
            trunc_normal_(self.head.weight, std=.02)

        self.apply(self._init_weights) # 应用相关的初始化权重

        if num_classes > 0:
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    #----------------------------#
    # 定义部分权重初始化操作
    #----------------------------#
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

    #-----------------------#
    # 涉及到的参数无权值衰减
    # 新添加lg_region_tokens
    #-----------------------#
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'part_tokens', 'lg_region_tokens'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head        = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    #-------------------------------#
    # 特征的前向传播函数
    #-------------------------------#
    def forward_features(self, x):

        x       = self.patch_embed(x) # 首先分成各个patch
        B, _, _ = x.size() # [B, N, C] # [20, 800, 512]
        # print(x.shape) # [20, 800, 512]

        #-----------------#
        # 位置嵌入并过dp
        #-----------------#
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach() # [20, 800, 512]

        x = self.pos_drop(x) # [B, N, C] # [20, 800, 512]

        #------------------------#
        # 根据注意力类型去展开计算
        #------------------------#
        if self.attn_type == 'local_global':
            #-------------------------------#
            # input: region partition
            # 根据设定的input值对张量进行切分
            #-------------------------------#
            nt, t = self.lg_num_region_size[0], self.lg_region_size[0] # nt=4, t=2
            nh, h = self.lg_num_region_size[1], self.lg_region_size[1] # nh=2, h=5
            nw, w = self.lg_num_region_size[2], self.lg_region_size[2] # nw=1, w=10

            b     = x.size(0)
            x     = rearrange(x, 'b (nt t nh h nw w) c -> b (nt nh nw) (t h w) c', nt=nt, nh=nh, nw=nw, t=t, h=h, w=w) # [20, 8, 100, 512]


            region_tokens = repeat(self.lg_region_tokens, 'n c -> b n 1 c', b=b) # [20, 8, 1, 512]

            x             = torch.cat([region_tokens, x], dim=2) # (b, nt*nh*nw, 1+thw, c) # [20, 8, 101, 512]

            x             = rearrange(x, 'b n s c -> (b n) s c') # s = 1 + thw # [160, 101, 512]
            

            intermediate_features = []
            for blk in self.blocks:
                # print(f'before blk, the shape is: {x.shape}') # [160, 101, 512] # correct
                x     = blk(x, b) # (b*n, s, c) # [160, 101, 512]
                x_new = rearrange(x, '(b n) s c -> b n s c', b=b) # s = 1 + thw [20, 8, 101, 512]

                if self.lg_classify_token_type   == 'region': # only use region tokens for classification  [默认为这种情况]
                    x_new = x_new[:,:,0] # (b, n, c) 相当于把第三维度给拿掉 # [20, 8, 512]

                elif self.lg_classify_token_type == 'org': # only use original tokens for classification
                    x_new = rearrange(x_new[:,:,1:], 'b n s c -> b (n s) c') # s = thw # [20, 8, 512]

                else: # use all tokens for classification # too heavy
                    x_new = rearrange(x_new, 'b n s c -> b (n s) c') # s = 1 + thw # [20, 808, 512]

                intermediate_features.append(x_new) # 当为region的方式时, x为[20, 8, 512] # 我们默认以region的方式向下走
            
        else:

            intermediate_features = []
            for blk in self.blocks:
                x = blk(x) # [20, 800, 512]
                intermediate_features.append(x) # x均为[20, 800, 512]

        x = torch.stack(intermediate_features, dim=2) # (B, N, L, D), L: number of layers, N: tokens


        x = self.norm(x) # [20, 8, 10, 512] # shape在norm前后不发生改变

        if self.fc_norm is not None:

            if self.keep_temporal_dim:
                x = rearrange(x, 'b (t hw) c -> b c t hw', t=self.patch_embed.temporal_seq_len, hw=self.patch_embed.spatial_num_patches)

                # spatial mean pooling
                x = x.mean(-1) # (B, C, T)

                # temporal upsample: 8 -> 16, for patch embedding reduction
                x = torch.nn.functional.interpolate(x, scale_factor=self.patch_embed.tubelet_size, mode='linear')

                x = rearrange(x, 'b c t -> b t c')

                return self.fc_norm(x)
            
            else:
                return self.fc_norm(x.mean(1)) # 使用mean pooling
        else:
            return x # 为此种情况 # [20, 8, 10, 512] # correct

    def forward(self, x):
        # print(x.shape) # [20, 3, 16, 160, 160]
        x = self.forward_features(x) # [20, 8, 10, 512]
        # print(f'after feature encoding, the shape is {x.shape}') # [20, 8, 10, 512]

        x = self.head(x) # 分类头 # 为空, 因此没有作用 # [20, 8, 10, 512]

        # me: add head activation function support, 用于支持回归任务
        x = self.head_activation_func(x) # 为空 # [20, 8, 10, 512]

        if self.keep_temporal_dim:
            x = x.view(x.size(0), -1) # (B,T,C) -> (B,T*C)

        return x # shape为 [B, N, L, C] # [20, 8, 10, 512]  L: number of layers, N: tokens number # correct [2024.6.30]


class VisionTransformerEncoder2D(nn.Module):
    """
    For Audio 
    Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_checkpoint=False,
                 use_learnable_pos_emb=False, init_scale=0., use_mean_pooling=True,
                 # me: new added [2024.6.26 20.54]
                 attn_type='joint', # 补充注意力类型的选择
                 lg_region_size_audio=(4, 4), lg_first_attn_type='self', lg_third_attn_type='cross',  # for local_global # 与视频分支的参数配置相比, 我们只有lg_region_size不一致, 其余均一致
                 lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False,
                 lg_classify_token_type='org', lg_no_second=False, lg_no_third=False,
                 ):
        super().__init__()

        self.num_classes    = num_classes
        self.num_features   = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed    = PatchEmbed2D(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=patch_size) # 进行通道数调整
        num_patches         = self.patch_embed.num_patches

        self.use_checkpoint = use_checkpoint

        #----------------------------#
        # 位置信息编码相关
        #----------------------------#
        if use_learnable_pos_emb:
            self.pos_embed  = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else: # NOTE: we deploy this manner
              # sine-cosine positional embeddings
            self.pos_embed  = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr           = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule


        self.attn_type     = attn_type
        #------------------------------#
        # 此处为不同的attention
        # 根据attn_type进行指定
        #------------------------------#
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
                  f"lg_no_second={lg_no_second}, lg_no_third={lg_no_third})") # 8个参数
            
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

            self.lg_region_size         = lg_region_size_audio # (h, w) # 指定的为(4, 4)
            print(f"The h and w of lg_region_size for AUDIO are {self.lg_region_size[0], self.lg_region_size[1]}")

            self.lg_num_region_size     = list(i//j for i,j in zip(self.patch_embed.input_token_size, lg_region_size_audio)) # (nh, nw) # 16, 8 分别除以 4, 4 得到的结果为4, 2 # 所以nh与nw的值分别为4与2
            num_regions                 = self.lg_num_region_size[0] * self.lg_num_region_size[1] # nh * nw
            print(f"==> The number of local regions for Auido is : {num_regions} (size={self.lg_num_region_size})") # region的数目为8 size为[4, 2]


            self.lg_region_tokens       = nn.Parameter(torch.zeros(num_regions, embed_dim)) # num_regions为8 shape为 [8, 512]
            trunc_normal_(self.lg_region_tokens, std=.02)

            self.lg_classify_token_type = lg_classify_token_type # 默认为region
            assert lg_classify_token_type in ['org', 'region', 'all'], f"Error: wrong 'lg_classify_token_type' in local_global attention ('{lg_classify_token_type}'), expected 'org'/'region'/'all'!"

        else:
            raise NotImplementedError
        

        self.norm    = nn.Identity() if use_mean_pooling else norm_layer(embed_dim) # 存在
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None # 默认为空

        self.head    = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity() # 为空

        if use_learnable_pos_emb: # no use
            trunc_normal_(self.pos_embed, std=.02)

        if num_classes > 0: # use
            trunc_normal_(self.head.weight, std=.02)

        self.apply(self._init_weights) # # 应用相关的初始化权重

        if num_classes > 0: # use
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    #-------------------------#
    # 定义部分权重的初始化操作
    #-------------------------#
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
        B, _, _ = x.size() # [B, N, C] # [20, 128, 512]

        if self.pos_embed is not None:
            x   = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()

        x       = self.pos_drop(x) # [20, 128, 512]

        #------------------------#
        # 根据注意力类型去展开计算
        #------------------------#
        if self.attn_type == 'local_global':

            nh, h = self.lg_num_region_size[0], self.lg_region_size[0] # nh=4, h=4
            nw, w = self.lg_num_region_size[1], self.lg_region_size[1] # nw=2, w=4

            b     = x.size(0) # [20, 128, 512]
            x     = rearrange(x, 'b (nh h nw w) c -> b (nh nw) (h w) c', nh=nh, nw=nw, h=h, w=w) # [20, 8, 16, 512]

            region_tokens = repeat(self.lg_region_tokens, 'n c -> b n 1 c', b=b) # [20, 8, 1, 512] # n为8 c为512

            x             = torch.cat([region_tokens, x], dim=2) # (b, nt*nh*nw, 1+thw, c) # [20, 8, 17, 512]

            x             = rearrange(x, 'b n s c -> (b n) s c') # s = 1 + thw # [160, 17, 512]
            

            intermediate_features = []
            for blk in self.blocks:
                # print(f'before blk, the shape is {x.shape}') # [160, 17, 512] # correct

                x     = blk(x, b) # (b*n, s, c) # [160, 17, 512] # LGBlock的forward中要求传入batch值
                x_new = rearrange(x, '(b n) s c -> b n s c', b=b) # s = 1 + thw [20, 8, 17, 512]

                if self.lg_classify_token_type   == 'region': # only use region tokens for classification  [默认为这种情况]
                    x_new = x_new[:,:,0] # (b, n, c) 相当于把第三维度给拿掉 # [20, 8, 512] # final USE

                elif self.lg_classify_token_type == 'org': # only use original tokens for classification
                    x_new = rearrange(x_new[:,:,1:], 'b n s c -> b (n s) c') # s = thw # [20, 8, 512]

                else: # use all tokens for classification
                    x_new = rearrange(x_new, 'b n s c -> b (n s) c') # s = 1 + thw # [20, 136, 512]

                intermediate_features.append(x_new) # 当为region的方式时, x为[20, 8, 512] # 我们默认以region的方式向下走

            # print(intermediate_features[0].shape) # [20, 8, 512]

        else:
            #------------------------------#
            # 音频模态也需要把中间张量存起来
            # 进入之前为[20, 128, 512]
            #------------------------------#
            intermediate_features = []
            for blk in self.blocks:
                x = blk(x) # [20, 128, 512]
                intermediate_features.append(x) # [20, 128, 512]


        x = torch.stack(intermediate_features, dim=2) # (B, N, L, D), L: number of layers, N: tokens  # [20, 8, 10, 512]
        # print(f'after stacking, the shape is {x.shape}') # [20, 8, 10, 512] # correct [2024.6.30]

        x = self.norm(x) # norm layer存在 # [20, 8, 10, 512]

        if self.fc_norm is not None: # NO USE!!
            # print('Using fc_norm!!') # 没有使用!!!
            return self.fc_norm(x.mean(1))
        else:
            return x # [20, 8, 10, 512] # 相当于与视频张量的shape保持一致 [2024.6.26 22.33]

    #-------------------------#
    # 整体前向传播函数
    #-------------------------#
    def forward(self, x):
        x = self.forward_features(x) # [20, 8, 10, 512]
        # print(f'after audio feature encoding, the feature shape is {x.shape}') # [20, 8, 10, 512]

        x = self.head(x) # 为空, 因此没有作用 # [20, 8, 10, 512]

        return x # 输出的维度与视频模态的保持一致 [B, N, L, C] # [20, 8, 10, 512] # correct [2024.6.30]


class VisionTransformerEncoderForFusion(nn.Module):
    """ 
    Audio-Visual Fusion Encoder
    """
    def __init__(self, embed_dim=768, embed_dim_audio=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,):
        super().__init__()

        dpr         = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule # 为下面的融合模块去服务

        self.blocks = nn.ModuleList([
            CSBlock(dim=embed_dim, context_dim=embed_dim_audio, num_heads=num_heads,
                    num_cross_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) # 视频norm层


        self.blocks_audio = nn.ModuleList([
            CSBlock(dim=embed_dim_audio, context_dim=embed_dim, num_heads=num_heads,
                    num_cross_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
            for i in range(depth)])
        self.norm_audio = norm_layer(embed_dim_audio) # do not share norm layer # 音频norm层


        self.apply(self._init_weights) # 应用初始化参数

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
        # print("In the fusion module")
        # print(f'input video shape {x.shape}') # [20, 8, 512]
        # print(f'audio input shape {x_audio.shape}') # [20, 8, 512]

        for blk, blk_audio in zip(self.blocks, self.blocks_audio):
            x, x_audio = blk(x, context=x_audio), blk_audio(x_audio, context=x)

        # print('After blks')
        # print(f'video shape {x.shape}') # [20, 8, 512] # correct [2024.6.30]
        # print(f'audio shape {x_audio.shape}') # [20, 8, 512] # correct # correct [2024.6.30]

        # norm layers
        # 各自过各自的norm layer
        x       = self.norm(x)
        x_audio = self.norm_audio(x_audio)

        return x, x_audio # shape与输入前的保持一致


class AudioVisionTransformer(nn.Module):
    """ 
    Audio-Visual Encoders
    """
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
                 all_frames=16, #全部的帧数 T维度
                 tubelet_size=2,
                 keep_temporal_dim=False, # do not use it for audio-visual
                 attn_type='joint',
                 img_size_audio=(256, 128),  # (T, F)
                 patch_size_audio=16,
                 in_chans_audio=1,
                 embed_dim_audio=768,
                 depth_audio=12,
                 num_heads_audio=12,
                 fusion_depth=2,
                 fusion_num_heads=12,
                 use_mean_pooling=False, # no use for now
                 head_activation_func=None,  # activation function after head fc, mainly for the regression task
                 # me: new added [2024.6.26 21.36]
                 lg_region_size=(2, 5, 10), # for video
                 lg_region_size_audio=(4, 4), # for audio
                 lg_first_attn_type='self', lg_third_attn_type='cross', # remain same
                 lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False, lg_classify_token_type='region', lg_no_second=False, lg_no_third=False, # remain same
                 ):
        super().__init__()

        # depthe of video encoder
        self.encoder_depth = depth

        if depth > 0 :
            self.encoder = VisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                num_classes=0, # no head for video encoder
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
                # me: new added [2024, 6.26, 21.40]
                lg_region_size=lg_region_size, # for video
                lg_first_attn_type=lg_first_attn_type, lg_third_attn_type=lg_third_attn_type, # remain same
                lg_attn_param_sharing_first_third=lg_attn_param_sharing_first_third, lg_attn_param_sharing_all=lg_attn_param_sharing_all, 
                lg_classify_token_type=lg_classify_token_type, lg_no_second=lg_no_second, lg_no_third=lg_no_third, # remain same
            )

        else:
            self.encoder = None
            print(f"==> Warning: video-specific encoder is not used!!!")


        # depthe of audio encoder
        self.encoder_depth_audio = depth_audio

        if depth_audio > 0:
            self.encoder_audio = VisionTransformerEncoder2D(
                img_size=img_size_audio,
                patch_size=patch_size_audio,
                in_chans=in_chans_audio,
                num_classes=0, # no head for audio encoder
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
                # me: new added [2024.6.26 21.40]
                attn_type=attn_type, # [6.30 11.56 new added] [大类定义传进来]
                lg_region_size_audio=lg_region_size_audio, # for audio
                lg_first_attn_type=lg_first_attn_type, lg_third_attn_type=lg_third_attn_type, # remain same
                lg_attn_param_sharing_first_third=lg_attn_param_sharing_first_third, lg_attn_param_sharing_all=lg_attn_param_sharing_all, 
                lg_classify_token_type=lg_classify_token_type, lg_no_second=lg_no_second, lg_no_third=lg_no_third, # remain same
            )

        else:
            self.encoder_audio = None
            print(f"==> Warning: audio-specific encoder is not used!!!")

        print(f'The embedding dims of video and audio encoders are {embed_dim} and {embed_dim_audio}, respectively.')
        print(f'The number of attention heads in video and audio encoders are {num_heads} and {num_heads_audio}, respectively.')
        print(f'The depths of video and audio encoders are {depth} and {depth_audio}, respectively.')

        if fusion_depth > 0 and self.encoder is not None and self.encoder_audio is not None:
            self.encoder_fusion = VisionTransformerEncoderForFusion_new_no_Mask_IR(
                embed_dim=embed_dim, # for video
                embed_dim_audio=embed_dim_audio, # for audio
                depth=fusion_depth,
                num_heads=fusion_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=0, # no drop path for fusion encoder
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

        # head
        fc_num       = embed_dim + embed_dim_audio    # 只有视听两部分张量进行融合, 因此两部分进行相加即可
        print(f'The final dims (fc_num) is {fc_num}')


        self.fc_norm = norm_layer(fc_num) # 在最后进行使用
        self.head    = nn.Linear(fc_num, num_classes) if num_classes > 0 else nn.Identity()
        print(f'The final dimension of head (classes) is {num_classes}')
        
        #---------------------------#
        # head的激活函数应用
        #---------------------------#
        if head_activation_func is not None:
            if head_activation_func == 'sigmoid':
                self.head_activation_func = nn.Sigmoid()
            elif head_activation_func == 'relu':
                self.head_activation_func = nn.ReLU()
            elif head_activation_func == 'tanh':
                self.head_activation_func = nn.Tanh()
            else:
                raise NotImplementedError
        else: # by default
            self.head_activation_func = nn.Identity()

        #---------------------------#
        # 对模态编码器的层数进行定义
        #---------------------------#
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
    
    #----------------------------------------------------------#
    # 编码器层数的最大值以及融合模块的层数, 相加则为模型的总层数
    #----------------------------------------------------------#
    def get_num_layers(self):
        # return len(self.blocks)
        return max(self.encoder_depth, self.encoder_depth_audio) + \
               self.encoder_depth_fusion

    def get_num_modality_specific_layers(self):
        # return len(self.blocks)
        return max(self.encoder_depth, self.encoder_depth_audio)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'part_tokens', 'lg_region_tokens',
                'encoder.pos_embed', 'encoder_audio.pos_embed', 'encoder_fusion.pos_embed',
                'encoder.cls_token', 'encoder_audio.cls_token', 'encoder_fusion.cls_token',}

    def forward_features(self, x, x_audio):

        # encoder: video
        x       = self.encoder(x) # [B, N, L, C] # [20, 8, 10, 512] # correct [2024.6.30]

        # encoder: audio
        x_audio = self.encoder_audio(x_audio) # [B, N, L, C] # [20, 8, 10, 512] # correct [2024.6.30]


        layer_weights_video = torch.nn.functional.softmax(self.layer_weights_video, dim=0)
        layer_weights_audio = torch.nn.functional.softmax(self.layer_weights_audio, dim=0)

        video_feature       = torch.einsum('bnlc,l->bnc', x, layer_weights_video)        # [B, N, C]
        audio_feature       = torch.einsum('bnlc,l->bnc', x_audio, layer_weights_audio)  # [B, N, C]

        overall_feat        = torch.cat([video_feature, audio_feature], dim=-1) # [B, N, 2C] # [20, 8, 1024] # 先video再audio

        x                    = x[:,:,-1]       # [B, N, C] # [20, 8, 10, 512] -> [20, 8, 512]
        x_audio              = x_audio[:,:,-1] # [B, N, C] # [20, 8, 10, 512] -> [20, 8, 512]


        x, x_audio, _  = self.encoder_fusion(x, x_audio, overall_feat)
        # print(f'after encoder fusion, the video is {x.shape}, the audio is {x_audio.shape}') # [bs, 8, C]


        # 这里是直接把第一维度给mean掉
        fusion_video_feature   = x.mean(dim=1)            # [B, C] # [20, 512]
        fusion_audio_feature   = x_audio.mean(dim=1)      # [B, C] # [20, 512]
        # print(f'after meaning, the video is {fusion_video_feature.shape}, the audio is {fusion_audio_feature.shape}') # [bs, C]

        final_feature          = torch.cat([fusion_video_feature, fusion_audio_feature], dim=-1) # 当前只有两部分堆叠, 输出为[20, 1024]
        # print(f'after the final concating, the final feature is {final_feature.shape}') # [BS, C] expected

        return self.fc_norm(final_feature) # 完成norm化的输出张量 # shape不变, 仍然为 [20, 1024]


    def forward(self, x, x_audio, save_feature=False):


        x = self.forward_features(x, x_audio) # (B, C) # [20, 1024] 视听张量堆叠的结果
        # print("#---------------------------------------------------------#")
        # print(f'After the feature encoding, the fusion shape is: {x.shape}')

        if save_feature:
            feature = x

        x = self.head(x)

        x = self.head_activation_func(x)

        if save_feature:
            return x, feature
        else:
            return x # (B, C) # final output [20, 6]


#--------------------------------#
# base version
#--------------------------------#
@register_model
def avit_dim512_patch16_160_a256(pretrained=False, **kwargs):
    embed_dim  = 512 # specific
    num_heads  = 8   # specific
    patch_size = 16

    model = AudioVisionTransformer(
        # video
        img_size=160,
        patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
        # audio
        img_size_audio=(256, 128),  # (T, F)
        patch_size_audio=patch_size, embed_dim_audio=embed_dim, num_heads_audio=num_heads,
        # fusion
        fusion_num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    model.default_cfg = _cfg()

    return model


#--------------------------------#
# large version
#--------------------------------#
@register_model
def avit_dim640_patch16_160_a256(pretrained=False, **kwargs):
    embed_dim  = 640 # specific
    num_heads  = 10  # specific
    patch_size = 16

    model = AudioVisionTransformer(
        # video
        img_size=160,
        patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
        # audio
        img_size_audio=(256, 128),  # (T, F)
        patch_size_audio=patch_size, embed_dim_audio=embed_dim, num_heads_audio=num_heads,
        # fusion
        fusion_num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    model.default_cfg = _cfg()

    return model


#--------------------------------#
# huge version
#--------------------------------#
@register_model
def avit_dim768_patch16_160_a256(pretrained=False, **kwargs):
    embed_dim  = 768 # specific
    num_heads  = 12  # specific
    patch_size = 16

    model = AudioVisionTransformer(
        # video
        img_size=160,
        patch_size=patch_size, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True,
        # audio
        img_size_audio=(256, 128),  # (T, F)
        patch_size_audio=patch_size, embed_dim_audio=embed_dim, num_heads_audio=num_heads,
        # fusion
        fusion_num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    model.default_cfg = _cfg()

    return model