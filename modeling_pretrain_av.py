import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from modeling_finetune_av import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table, CSBlock, PatchEmbed2D, LGBlock
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from einops import rearrange, repeat
from modeling_attention_av import PretrainVisionTransformerEncoderForFusion_new_no_Mask_IR # [用于整体预训练模型]

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PretrainVisionTransformerEncoder(nn.Module):
    """ 
    Video Encoder.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2,
                 use_learnable_pos_emb=False,
                 # me: new added [2024.6.29 15.45]
                 attn_type='joint',
                 lg_region_size=(2, 5, 10), lg_first_attn_type='self', lg_third_attn_type='cross',  # for local_global # 此为新添加的
                 lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False,
                 lg_no_second=False, lg_no_third=False,
                 ):
        
        super().__init__()
        self.num_classes  = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed  = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, tubelet_size=tubelet_size) # 正常的patch embed 为面向三维的
        num_patches       = self.patch_embed.num_patches

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # NOTE: our choice
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)


        dpr         = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule # 每个block的dropout rate


        self.attn_type  = attn_type # 新增参数

        if attn_type   == 'joint':
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
                for i in range(depth)])

        elif attn_type == 'local_global':

            # 会打印输出一系列的信息
            print(f"==> Note for VIDEO: Use 'local_global' for compute reduction (lg_region_size={lg_region_size},"
                  f"lg_first_attn_type={lg_first_attn_type}, lg_third_attn_type={lg_third_attn_type},"
                  f"lg_attn_param_sharing_first_third={lg_attn_param_sharing_first_third},"
                  f"lg_attn_param_sharing_all={lg_attn_param_sharing_all},"
                  f"lg_no_second={lg_no_second}, lg_no_third={lg_no_third})") # 共7个参数
            
            #---------------------------#
            # 基于LGBlock进行构建
            # 共有depth个
            #---------------------------#
            self.blocks = nn.ModuleList([
                LGBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values,
                    first_attn_type=lg_first_attn_type, third_attn_type=lg_third_attn_type,
                    attn_param_sharing_first_third=lg_attn_param_sharing_first_third,
                    attn_param_sharing_all=lg_attn_param_sharing_all,
                    no_second=lg_no_second, no_third=lg_no_third,) 
                for i in range(depth)]) # 这里使用了除lg_region_size外的所有参数
            
            #---------------------#
            # 区域tokens
            #---------------------#
            self.lg_region_size     = lg_region_size # (t, h, w) # (2, 5, 10) 这是在sh中指定的
            print(f'The t, h, and w of lg_region_size for VIDEO: {self.lg_region_size[0]}, {self.lg_region_size[1]}, {self.lg_region_size[2]}')

            self.lg_num_region_size = list(i//j for i,j in zip(self.patch_embed.input_token_size, lg_region_size)) # (nt, nh, nw) 8 10 10 分别除以 2 5 10 得到结果为 4 2 1
            num_regions             = self.lg_num_region_size[0] * self.lg_num_region_size[1] * self.lg_num_region_size[2] # nt * nh * nw # 相当于为8个region
            print(f"==> The number of local regions: {num_regions} (size={self.lg_num_region_size})") # The number of local regions: 8 (size=[4, 2, 1])

            #-----------------------------#
            # 额外定义的tokens
            #-----------------------------#
            self.lg_region_tokens   = nn.Parameter(torch.zeros(num_regions, embed_dim))  # [8, 512]
            trunc_normal_(self.lg_region_tokens, std=.02)
            # print(self.lg_region_tokens.shape) # [8, 512]

        else:
            raise NotImplementedError


        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity() # 为空

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights) # 利用初始化权重的函数进行初始化



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
        return {'pos_embed', 'cls_token', 'part_tokens', 'lg_region_tokens'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head        = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x, mask, return_intermediate_features=None):
        _, _, T, _, _ = x.shape 

        x             = self.patch_embed(x)
        # print(f'after path embed, the tensor is {x.shape}') # [20, 800, 512]

        x             = x + self.pos_embed.type_as(x).to(x.device).clone().detach() # B, N, C 三维
        # print(f'after residual, the tensor is {x.shape}') # [20, 800, 512]

        B, _, C       = x.shape
        # print(x.shape) # [20, 800, 512]

        x_vis         = x[~mask].reshape(B, -1, C) 


        if self.attn_type == 'local_global':
            
            unmask_ratio   = x_vis.size(1) / x.size(1)
            # print(unmask_ratio) # 0.1 相当于为80/800 进而得出为0.1

            nt, t          = self.lg_num_region_size[0], self.lg_region_size[0] # self.lg_num_region_size[0]=4 self.lg_region_size[0]=2
            # print(f'nt {nt}') # nt=4
            # print(f't {t}') # t=2

            nhw            = self.lg_num_region_size[1] * self.lg_num_region_size[2] # self.lg_num_region_size[1]=2 self.lg_num_region_size[2]=1
            # print(f'nhw {nhw}') nhw=2

            hw             = int(self.lg_region_size[1] * self.lg_region_size[2] * unmask_ratio) # self.lg_region_size[1]=5 self.lg_region_size[2]=10  # unmask_ratio=0.1
            # print(f'hw {hw}') hw=5

            b              = x_vis.size(0) # b=20
            # print(f'b {b}') b=64 相当于为batch size

            x_vis          = rearrange(x_vis, 'b (nt t nhw hw) c -> b (nt nhw) (t hw) c', nt=nt, t=t, nhw=nhw, hw=hw) # [20, 8, 10, 512]
            # print('after rearrange: ')
            # print(x_vis.shape) # [20, 8, 10, 512]

            # add region tokens
            region_tokens  = repeat(self.lg_region_tokens, 'n c -> b n 1 c', b=b) # [20, 8, 1, 512]
            # print(region_tokens.shape)

            x_vis          = torch.cat([region_tokens, x_vis], dim=2) # (b, nt*nh*nw, 1+thw, c) # [20, 8, 11, 512] # 相当于是在最前面加了一个token
            # print(f'after cat, the x_vis shape is {x_vis.shape}')

            x_vis          = rearrange(x_vis, 'b n s c -> (b n) s c') # s = 1 + thw # 这里的形式仍然符合LGBlock输入的要求 # [160, 11, 512]
            # print(f'the final shape of x_vis is {x_vis.shape}')


            intermediate_features = []
            for blk in self.blocks:
                x_vis  = blk(x_vis, b) # (b*n, s, c) # [160, 11, 512]

                x_vis_a = rearrange(x_vis[:,1:], '(b n) s c -> b (n s) c', b=b) # [20, 80, 512]
                intermediate_features.append(self.norm(x_vis_a)) # 均为[20, 80, 512]

            x_vis = rearrange(x_vis[:,1:], '(b n) s c -> b (n s) c', b=b) # s = thw # 本质上的整体shape还是没有发生改变的 # x_vis[:,1:]即为将第一个token去除(这是后面加的, 需要去除, 从而保持原始tokens)
            # print(f'after blks, the shape is {x_vis.shape}') # [20, 80, 512]

        else:
            intermediate_features = []
            for blk in self.blocks:
                x_vis = blk(x_vis) # 先过norm, 再添加到列表中去
                # print(f'during blocks, the shape is {x_vis.shape}') # [20, 80, 512], 共有10个
                intermediate_features.append(self.norm(x_vis))

        #-------------------------------#
        # 对最后输出的特征进行norm化操作
        #-------------------------------#
        x_vis = self.norm(x_vis)

        if return_intermediate_features is None:
            return x_vis, None
        
        else:
            intermediate_features = [intermediate_features[i] for i in return_intermediate_features] # return_intermediate_features可能为(3 6 9) 那就输出对应层的特征即可
            return x_vis, intermediate_features

    #-----------------#
    # 总体前向传播函数
    #-----------------#
    def forward(self, x, mask, return_intermediate_features=None):
        x, intermediate_features = self.forward_features(x, mask, return_intermediate_features)

        x                        = self.head(x) # 这一层相当于是不作用的 nn.Identity()
        # print(f'The final shape is {x.shape}') # [20, 80, 512]

        return x, intermediate_features # 均为[20, 80, 512] # 80为tokens的数目


class PretrainVisionTransformerEncoder2D(nn.Module):
    """ 
    Audio Encoder.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_checkpoint=False,
                 use_learnable_pos_emb=False,
                 # me: new added [2024.6.29 16.21]
                 attn_type='joint', # 补充注意力类型的选择
                 lg_region_size_audio=(4, 4), lg_first_attn_type='self', lg_third_attn_type='cross',
                 lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False,
                 lg_no_second=False, lg_no_third=False,
                 ):
        super().__init__()

        self.num_classes    = num_classes
        self.num_features   = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed    = PatchEmbed2D(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=patch_size) # 用于2D的位置信息嵌入
        num_patches         = self.patch_embed.num_patches

        self.use_checkpoint = use_checkpoint

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else: # NOTE: we deploy this manner
              # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule # 每个block的dropout rate


        self.attn_type     = attn_type
        if attn_type      == 'joint':
            self.blocks    = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
                for i in range(depth)])
        

        elif attn_type == 'local_global':
            print(f"==> Note for AUDIO: Use 'local_global' for compute reduction (lg_region_size_audio={lg_region_size_audio},"
                  f"lg_first_attn_type={lg_first_attn_type}, lg_third_attn_type={lg_third_attn_type},"
                  f"lg_attn_param_sharing_first_third={lg_attn_param_sharing_first_third},"
                  f"lg_attn_param_sharing_all={lg_attn_param_sharing_all},"
                  f"lg_no_second={lg_no_second}, lg_no_third={lg_no_third})") # 共7个参数
            

            self.blocks = nn.ModuleList([
                LGBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values,
                    first_attn_type=lg_first_attn_type, third_attn_type=lg_third_attn_type,
                    attn_param_sharing_first_third=lg_attn_param_sharing_first_third,
                    attn_param_sharing_all=lg_attn_param_sharing_all,
                    no_second=lg_no_second, no_third=lg_no_third,) 
                for i in range(depth)])
            
            self.lg_region_size     = lg_region_size_audio
            print(f"The h and w of lg_region_size_audio for AUDIO are {self.lg_region_size[0], self.lg_region_size[1]}")

            self.lg_num_region_size = list(i//j for i,j in zip(self.patch_embed.input_token_size, lg_region_size_audio))
            num_regions             = self.lg_num_region_size[0] * self.lg_num_region_size[1] # nh * nw # 相当于为8个region
            print(f"==> The number of local regions for AUDIO is: {num_regions} (size={self.lg_num_region_size})") # The number of local regions: 8 (size=[4, 2])


            self.lg_region_tokens   = nn.Parameter(torch.zeros(num_regions, embed_dim))  # [8, 512]
            trunc_normal_(self.lg_region_tokens, std=.02)

        else:
            raise NotImplementedError


        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity() # 默认为空

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights) # 利用初始化权重的函数进行初始化



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
        return {'pos_embed', 'cls_token', 'part_tokens', 'lg_region_tokens'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head        = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x, mask, return_intermediate_features=None):
        # print(f'input audio shape: {x.shape}') # [20, 1, 256, 128]

        x = self.patch_embed(x) # 为2D的patch embed
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach() # 三维

        B, _, C = x.shape # [B, N, C]
        # print(x.shape) # [20, 128, 512]

        x_vis   = x[~mask].reshape(B, -1, C)  # "~mask" means visible
        # print(x_vis.shape) # [20, 24, 512] # 也是对布尔张量取反后的结果

        if self.attn_type == 'local_global':
            
            # print("#local_global#")
            # input: region partition
            # print(x_vis.size(1)) # 24
            # print(x.size(1)) # 128

            unmask_ratio   = x_vis.size(1) / x.size(1) # 为0.1875 NOTE: 对小数不进行保留
            # print(unmask_ratio) # 0.1875

            nhw            = self.lg_num_region_size[0] * self.lg_num_region_size[1] # self.lg_num_region_size[0]=4 self.lg_num_region_size[1]=2
            # print(f'nhw {nhw}') nhw=8

            hw             = int(self.lg_region_size[0] * self.lg_region_size[1] * unmask_ratio) # self.lg_region_size[0]=4 self.lg_region_size[1]=4  # unmask_ratio=0.1875 # 所以hw直接为整数3
            # print(f'hw {hw}') # hw=3 # 这里hw直接为整数3

            b              = x_vis.size(0) # b=20
            # print(f'b {b}') b=64 相当于为batch size

            x_vis          = rearrange(x_vis, 'b (nhw hw) c -> b (nhw) (hw) c', nhw=nhw, hw=hw) # nhw为8, hw为3, 转变后为[20, 8, 3, 512]
            # print('after rearrange: ')
            # print(x_vis.shape) # [20, 8, 3, 512]

            # add region tokens
            region_tokens  = repeat(self.lg_region_tokens, 'n c -> b n 1 c', b=b) # [20, 8, 1, 512]
            # print(region_tokens.shape)

            x_vis          = torch.cat([region_tokens, x_vis], dim=2) # (b, nh*nw, 1+hw, c) # [20, 8, 4, 512] # 相当于是在最前面加了一个token
            # print(f'after cat, the x_vis shape is {x_vis.shape}')

            x_vis          = rearrange(x_vis, 'b n s c -> (b n) s c') # s = 1 + thw # 这里的形式仍然符合LGBlock输入的要求 # [160, 4, 512]
            # print(f'the final shape of x_vis is {x_vis.shape}')


            intermediate_features = []
            for blk in self.blocks:
                x_vis  = blk(x_vis, b) # (b*n, s, c) # [160, 4, 512]

                x_vis_a = rearrange(x_vis[:,1:], '(b n) s c -> b (n s) c', b=b)  # [20, 24, 512]
                # print(x_vis_a.shape)

                intermediate_features.append(self.norm(x_vis_a)) # 保存到列表中的均为[20, 24, 512]

            # print(intermediate_features[0].shape) # [20, 24, 512]
            # keep only use original tokens for decoder
            x_vis = rearrange(x_vis[:,1:], '(b n) s c -> b (n s) c', b=b) # s = thw # 本质上的整体shape还是没有发生改变的 # x_vis[:,1:]即为去掉第一个元素, 在完成自注意力作用后将第一个token给去除
            # print(f'after blks, the shape is {x_vis.shape}') # [20, 24, 512]
        
        else:
            intermediate_features = []
            #-------------------------#
            # 与上面的定义形式一致
            # 对中间张量进行保存
            #-------------------------#
            for blk in self.blocks:
                x_vis = blk(x_vis)
                # print(x_vis.shape) # [20, 24, 512] shape并不改变
                intermediate_features.append(self.norm(x_vis)) # [20, 24, 512]

        #------------------------------#
        # 对于最后的输出再过下norm layer
        #------------------------------#
        x_vis = self.norm(x_vis) # [20, 24, 512]

        if return_intermediate_features is None:
            return x_vis, None
        
        else:
            intermediate_features = [intermediate_features[i] for i in return_intermediate_features]
            return x_vis, intermediate_features # shape均为[20, 24, 512]

    #---------------------------------#
    # 整体前向传播函数的定义
    #---------------------------------#
    def forward(self, x, mask, return_intermediate_features=None):
        x, intermediate_features = self.forward_features(x, mask, return_intermediate_features)
        # print(f'after encoding, the tensor shape is {x.shape}') # 此为经过blocks作用后的最后一个张量, 其shape为[20, 24, 512]
        # print(f'the intermediate features are {intermediate_features[0].shape, intermediate_features[1].shape, intermediate_features[2].shape}') # 三个中间张量的shape也是[20, 24, 512]

        x                        = self.head(x) # 为空
        # print(x.shape) # [20, 24, 512]

        return x, intermediate_features # shape均为 [20, 24, 512] # 24为可见的tokens数目



class PretrainVisionTransformerEncoderForFusion(nn.Module):
    """ 
    For Audio-Visual Fusion.
    """
    def __init__(self, embed_dim=768, embed_dim_audio=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 modal_param_sharing=False):
        super().__init__()

        dpr         = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        #--------------------------------#
        # 视频相关
        # 用于融合的模块定义
        #--------------------------------#
        self.blocks = nn.ModuleList([
            CSBlock(dim=embed_dim, context_dim=embed_dim_audio, num_heads=num_heads,
                    num_cross_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) # for video

        #------------#
        # audio分支
        #------------#
        self.modal_param_sharing = modal_param_sharing


        if not modal_param_sharing:
            self.blocks_audio = nn.ModuleList([
                CSBlock(dim=embed_dim_audio, context_dim=embed_dim, num_heads=num_heads,
                        num_cross_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[i], norm_layer=norm_layer,
                        init_values=init_values)
                for i in range(depth)])
        else:
            self.blocks_audio = self.blocks # 如果modal_param_sharing为True, 则视频与音频的模块保持一致, 相当于进行参数共享
        
        #--------------------------#
        # do not share norm layer
        # 这个是音频模态特定的
        #--------------------------#
        self.norm_audio = norm_layer(embed_dim_audio) # for audio

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
        
        # print(f'input video shape: {x.shape}')       # [20, 80, 512]
        # print(f'input audio shape: {x_audio.shape}') # [20, 24, 512]

        #------------------------#
        # 此处均为CSBlock
        #------------------------#
        for blk, blk_audio in zip(self.blocks, self.blocks_audio):
            x, x_audio = blk(x, context=x_audio), blk_audio(x_audio, context=x)

        # print(x.shape)       # [20, 80, 512]
        # print(x_audio.shape) # [20, 24, 512]

        # through norm layers
        x       = self.norm(x)
        x_audio = self.norm_audio(x_audio)

        # print(x.shape)       # [20, 80, 512]
        # print(x_audio.shape) # [20, 24, 512]

        return x, x_audio
    
class PretrainVisionTransformerDecoder(nn.Module):
    """ 
    Pre-train Decoder for Video and Audio.
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2,):
        super().__init__()

        self.num_classes    = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2 # 对于音频而言就是patch_size ** 2 = 16 * 16 = 256

        self.num_features   = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size     = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        #-------------------------------#
        # 开始block的定义
        # 第一层是Block, 其余均为CSBlock
        #-------------------------------#
        self.blocks = nn.ModuleList([])

        for i in range(depth):
            if i == 0:
                self.blocks.append(
                    Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        init_values=init_values)
                )
            else:
                self.blocks.append(
                    CSBlock(
                        dim=embed_dim, context_dim=embed_dim, num_heads=num_heads, num_cross_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        init_values=init_values)
                )

        #--------------------#
        # 定义norm层和分类器
        #--------------------#
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity() # 此为用于通道数调整, 因此不为空 # 它们调整后的维度值与模态encoder中的num_patches有关

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


    def forward(self, x, return_token_num, x_skip_connects=None):

        for i, blk in enumerate(self.blocks):
            if i == 0:
                x = blk(x) # 普通的网络
            else: # hierarchical skip connections
                x = blk(x, context=x_skip_connects[-i]) # 第i=1层与第i=-1层进行融合, 注定decoder的层数为偶数


        # print(x.shape) # video: [20, 480, 384] audio: [20, 128, 384]
        if return_token_num > 0:
            # print('return_token_num > 0')
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels # x[:, -return_token_num:]的含义为选取最后return_token_num个tokens 然后进行归一化以及映射从而完成shape的调整
            # print(x.shape) # video: [20, 400, 1536] audio: [20, 104, 256]  # 注意在模型主体类中, 存在video与audio的decoder定义, 进而它们的head定义均不相同
        else:
            x = self.head(self.norm(x))
            print(f'when return_token_num <= 0, the shape is {x.shape}')

        return x # 最终输出



class PretrainAudioVisionTransformer(nn.Module):
    """ 
    The Pre-train Audio-Visual Transformer.
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=1536,  #  decoder_num_classes=768,
                 decoder_embed_dim=512,
                 decoder_depth=8, # 为4
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=2,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 # audio
                 img_size_audio=(256, 128),  # (T, F)
                 patch_size_audio=16,
                 encoder_in_chans_audio=1,
                 encoder_embed_dim_audio=768,
                 encoder_depth_audio=12,
                 encoder_num_heads_audio=12,
                 decoder_num_classes_audio=256,
                 decoder_embed_dim_audio=512,
                 decoder_depth_audio=8, # 为4
                 decoder_num_heads_audio=8,
                 # fusion
                 encoder_fusion_depth=2,
                 encoder_fusion_num_heads=12,
                 # contrastive learning
                 inter_contrastive_temperature=0.07,
                 # intermediate layers
                 return_intermediate_features=(3, 6, 9), # 为3 相当于是返回这些层的特征 即为第 4 7 10层的特征
                 # me: new added [2024.6.29 17.17]
                 attn_type='joint',
                 lg_region_size=(2, 5, 10), # for video
                 lg_region_size_audio=(4, 4), # for audio
                 lg_first_attn_type='self', lg_third_attn_type='cross', # remain same
                 lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False, lg_no_second=False, lg_no_third=False, # remain same
                 ):
        super().__init__()

        self.encoder_depth                = encoder_depth
        self.return_intermediate_features = return_intermediate_features
        print(f'The returned feats are: {self.return_intermediate_features}')


        if self.return_intermediate_features is not None:
            assert len(self.return_intermediate_features) == decoder_depth - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth-1}) for intermediate_features!"
            assert len(self.return_intermediate_features) == decoder_depth_audio - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth_audio-1}) for intermediate_features!"

            for idx in self.return_intermediate_features:
                assert idx < encoder_depth and idx < encoder_depth_audio, f"Error: too large layer index ({idx})!"

        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            # me: new added [2024.6.29 17.19]
            attn_type=attn_type,
            lg_region_size=lg_region_size, # for video
            lg_first_attn_type=lg_first_attn_type, lg_third_attn_type=lg_third_attn_type, # remain same
            lg_attn_param_sharing_first_third=lg_attn_param_sharing_first_third, lg_attn_param_sharing_all=lg_attn_param_sharing_all, lg_no_second=lg_no_second, lg_no_third=lg_no_third, # remain same
        )
        print(f'During Pre-train, the depths and attention heads of VIDEO Encoder are {encoder_depth}, {encoder_num_heads}, and the embed dims is {encoder_embed_dim}')

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
        )
        print(f'During Pre-train, the depths and attention heads of VIDEO Decoder are {decoder_depth}, {decoder_num_heads}, and the embed dims is {decoder_embed_dim}')

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.mask_token         = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed          = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

        self.encoder_audio = PretrainVisionTransformerEncoder2D(
            img_size=img_size_audio,
            patch_size=patch_size_audio,
            in_chans=encoder_in_chans_audio,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim_audio,
            depth=encoder_depth_audio,
            num_heads=encoder_num_heads_audio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
            # me: new added [2024.6.29 17.21]
            attn_type=attn_type,
            lg_region_size_audio=lg_region_size_audio, # for audio
            lg_first_attn_type=lg_first_attn_type, lg_third_attn_type=lg_third_attn_type, # remain same
            lg_attn_param_sharing_first_third=lg_attn_param_sharing_first_third, lg_attn_param_sharing_all=lg_attn_param_sharing_all, lg_no_second=lg_no_second, lg_no_third=lg_no_third, # remain same
        )
        print(f'During Pre-train, the depths and attention heads of AUDIO encoder are {encoder_depth_audio}, {encoder_num_heads_audio}, and the embed dims is {encoder_embed_dim_audio}')


        self.decoder_audio = PretrainVisionTransformerDecoder(
            patch_size=patch_size_audio,
            num_patches=self.encoder_audio.patch_embed.num_patches,
            num_classes=decoder_num_classes_audio,
            embed_dim=decoder_embed_dim_audio,
            depth=decoder_depth_audio,
            num_heads=decoder_num_heads_audio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=1/3, # no meaning, just to escape 'assert'
        )
        print(f'During Pre-train, the depths and attention heads of AUDIO Decoder are {decoder_depth_audio}, {decoder_num_heads_audio}, and the embed dims is {decoder_embed_dim_audio}')


        self.encoder_to_decoder_audio = nn.Linear(encoder_embed_dim_audio, decoder_embed_dim_audio, bias=False)
        self.mask_token_audio         = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim_audio))
        self.pos_embed_audio          = get_sinusoid_encoding_table(self.encoder_audio.patch_embed.num_patches, decoder_embed_dim_audio)

        trunc_normal_(self.mask_token_audio, std=.02)



        self.encoder_fusion = PretrainVisionTransformerEncoderForFusion_new_no_Mask_IR(
            embed_dim=encoder_embed_dim, # for video
            embed_dim_audio=encoder_embed_dim_audio, # for audio
            depth=encoder_fusion_depth,
            num_heads=encoder_fusion_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
        )
        print(f'During Pre-train, the depths and attention heads of fusion module are {encoder_fusion_depth}, {encoder_fusion_num_heads}, and the embed dims of video and audio are {encoder_embed_dim} and {encoder_embed_dim_audio}, respectively.')

        #------------------------#
        # 模态间对比学习的温度因子
        #------------------------#
        self.inter_contrastive_temperature = inter_contrastive_temperature

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
        return {'pos_embed', 'cls_token', 'mask_token', 'lg_region_tokens'}


    def forward(self, x, mask, decode_mask, x_audio, mask_audio, decode_mask_audio):

        # print(f'input video tensor shape is {x.shape}') # [20, 3, 16, 160, 160]
        # print(f'input audio tensor shape is {x_audio.shape}') # [20, 1, 256, 128]

        # print(f'The shape of mask is {mask.shape}') # [20, 800]
        # print(f'The shape of decode_mask is {decode_mask.shape}') # [20, 800]

        # print(f'The shape of mask_audio is {mask_audio.shape}') # [20, 128]
        # print(f'The shape of decode_mask_audio is {decode_mask_audio.shape}') # [20, 128]

        decode_vis       = mask if decode_mask is None else ~decode_mask # 表明如果decode_mask为0, 则为mask, 进而与原始VideoMAE保持一致
        decode_vis_audio = mask_audio if decode_mask_audio is None else ~decode_mask_audio # 表明如果decode_mask_audio为0, 则为mask_audio, 进而与原始VideoMAE保持一致

        # print(f'The shape of decode_vis is {decode_vis.shape}') # [20, 800]
        # print(f'The shape of decode_vis_audio is {decode_vis_audio.shape}') # [20, 128]

        # encoder: video
        x_vis, x_vis_inter_features             = self.encoder(x, mask, self.return_intermediate_features) # [B, N_vis, C_e] # [20, 80, 512]

        # encoder: audio
        x_vis_audio, x_vis_audio_inter_features = self.encoder_audio(x_audio, mask_audio, self.return_intermediate_features) # [20, 24, 512]


        logits_per_video, logits_per_audio = [], []
        for x_vis_inter, x_vis_audio_inter in zip(x_vis_inter_features, x_vis_audio_inter_features):
            # pooling
            video_features_inter = x_vis_inter.mean(dim=1)        # (B, C)
            audio_features_inter = x_vis_audio_inter.mean(dim=1)  # (B, C)

            # normalized features
            video_features_inter = video_features_inter / video_features_inter.norm(dim=1, keepdim=True)
            audio_features_inter = audio_features_inter / audio_features_inter.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logits_per_video_inter = video_features_inter @ audio_features_inter.t() / self.inter_contrastive_temperature # 温度因子在这里使用 # 通过矩阵乘法来构建点积, 从而衡量相似度, 再使用温度因子控制分数的范围
            logits_per_audio_inter = logits_per_video_inter.t()

            logits_per_video.append(logits_per_video_inter) # 直接return
            logits_per_audio.append(logits_per_audio_inter) # 直接return


        video_feature_1      = x_vis.mean(dim=1, keepdim=True)       # [20, 80, 512] -> [20, 1, 512]
        audio_feature_1      = x_vis_audio.mean(dim=1, keepdim=True) # [20, 24, 512] -> [20, 1, 512]
        overall_feat         = torch.cat([video_feature_1, audio_feature_1], dim=-1)  # [20, 1, 1024]

        x_vis, x_vis_audio, _, _ = self.encoder_fusion(x_vis, x_vis_audio, overall_feat) # 将三个张量送进去, 通过跨注意力实现了对原始张量的利用以及视音频特征的补充


        x_vis       = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d] # 维度调整
        # print(f'after encoder to decoder, the shape of x_vis for video is {x_vis.shape}') # [20, 80, 384]

        B, N_vis, C = x_vis.shape
        
        # we don't unshuffle the correct visible token order, but shuffle the pos embedding accordingly.
        expand_pos_embed     = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        # print(f'The shape of expand_pos_embed for video is {expand_pos_embed.shape}') # [20, 800, 384]

        pos_emd_vis          = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask         = expand_pos_embed[decode_vis].reshape(B, -1, C) # NOTE: 主要在这里存在很大的不同, 原先就是拿mask掉的token过来

        # print(pos_emd_vis.shape)  # [20, 80, 384]  # 没有被遮掩的
        # print(pos_emd_mask.shape) # [20, 400, 384] # 被遮掩的

        # x_full为完整的张量
        x_full               = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)  # [B, N, C_d]
        # print(f'The shape of x_full for video is {x_full.shape}') # [20, 480, 384] 将两部分张量合并起来作为整体

        x_vis_inter_features = [self.encoder_to_decoder(feature) + pos_emd_vis for feature in x_vis_inter_features]


        x                    = self.decoder(x_full, pos_emd_mask.shape[1], x_vis_inter_features)  # [B, N_mask, 2 * 3 * 16 * 16] # x 即为video decoder的最终输出
        # print(f'after decoder, the shape of x for video is {x.shape}') # [20, 400, 1536] 输出后用于在loss中进行MAE均值的计算 1536=2 * 3 * 16 * 16

        #----------------------------#
        # decoder: audio
        # 音频解码器, 其流程与video一致
        #----------------------------#
        x_vis_audio                = self.encoder_to_decoder_audio(x_vis_audio) # [B, N_vis, C_d]
        # print(f'after encoder to decoder, the shape of x_vis_audio for audio is {x_vis_audio.shape}') # [20, 24, 384]

        B_audio, N_audio, C_audio  = x_vis_audio.shape

        # we don't unshuffle the correct visible token order, but shuffle the pos embedding accordingly.
        expand_pos_embed_audio     = self.pos_embed_audio.expand(B_audio, -1, -1).type_as(x_audio).to(x_audio.device).clone().detach()
        # print(f'The shape of expand_pos_embed_audio is {expand_pos_embed_audio.shape}') # [20, 128, 384]

        pos_emd_vis_audio          = expand_pos_embed_audio[~mask_audio].reshape(B_audio, -1, C_audio)
        pos_emd_mask_audio         = expand_pos_embed_audio[decode_vis_audio].reshape(B_audio, -1, C_audio) # NOTE: 主要在这里存在很大的不同, 原先就是拿mask掉的token过来


        x_full_audio               = torch.cat([x_vis_audio + pos_emd_vis_audio, self.mask_token_audio + pos_emd_mask_audio], dim=1) # [B, N, C_d]


        x_vis_audio_inter_features = [self.encoder_to_decoder_audio(feature) + pos_emd_vis_audio for feature in x_vis_audio_inter_features]


        x_audio                    = self.decoder_audio(x_full_audio, pos_emd_mask_audio.shape[1], x_vis_audio_inter_features) # [B, N_mask, 1 * 16 * 16] # x_audio即为audio decoder的最终输出


        return x, x_audio, logits_per_video, logits_per_audio


@register_model
def pretrain_hicmae_dim512_patch16_160_a256(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformer(
        img_size=160,
        patch_size=16,
        encoder_embed_dim=512, # specific
        encoder_num_heads=8, # specific
        encoder_num_classes=0,
        decoder_num_classes=1536, # 16 * 16 * 3 * 2 (保持不变)
        decoder_embed_dim=384, # specific
        decoder_num_heads=6, # specific
        # audio
        img_size_audio=(256, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=512, # specific
        encoder_num_heads_audio=8, # specific
        decoder_num_classes_audio=256, # 16 * 16 (保持不变)
        decoder_embed_dim_audio=384, # specific
        decoder_num_heads_audio=6, # specific
        # fusion
        encoder_fusion_num_heads=8, # specific
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def pretrain_hicmae_dim640_patch16_160_a256(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformer(
        img_size=160,
        patch_size=16,
        encoder_embed_dim=640, # specific
        encoder_num_heads=10, # specific
        encoder_num_classes=0,
        decoder_num_classes=1536, # 16 * 16 * 3 * 2 (保持不变)
        decoder_embed_dim=512, # specific
        decoder_num_heads=8, # specific
        # audio
        img_size_audio=(256, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=640, # specific
        encoder_num_heads_audio=10, # specific
        decoder_num_classes_audio=256, # 16 * 16 (保持不变)
        decoder_embed_dim_audio=512, # specific
        decoder_num_heads_audio=8, # specific
        # fusion
        encoder_fusion_num_heads=10, # specific
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def pretrain_hicmae_dim768_patch16_160_a256(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformer(
        img_size=160,
        patch_size=16,
        encoder_embed_dim=768, # specific
        encoder_num_heads=12, # specific
        encoder_num_classes=0,
        decoder_num_classes=1536, # 16 * 16 * 3 * 2 (保持不变)
        decoder_embed_dim=640, # specific
        decoder_num_heads=8, # specific
        # audio
        img_size_audio=(256, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=768, # specific
        encoder_num_heads_audio=12, # specific
        decoder_num_classes_audio=256, # 16 * 16 (保持不变)
        decoder_embed_dim_audio=640, # specific
        decoder_num_heads_audio=8, # specific
        # fusion
        encoder_fusion_num_heads=12, # specific
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    return model