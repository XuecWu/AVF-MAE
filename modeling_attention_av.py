import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path

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
        self.act        = act_layer() # 此处进行初始化运用
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
        
        #---------------------------#
        # me: support window mask
        #---------------------------#
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
        q        = q.reshape(B, T1, self.num_heads, -1).transpose(1, 2) # me: (B, H, T1, C//H)

        kv       = F.linear(input=x if context is None else context, weight=self.kv.weight, bias=kv_bias) # 这里实现对于context的使用, 如果是自注意力则context为None, 如果是跨模态注意力则context非None

        _, T2, _ = kv.shape
        kv       = kv.reshape(B, T2, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v     = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple), me： (B, H, T2, C//H)


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


class IterativeRefinement(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, concat_dim=None):
        super().__init__()

        self.embedding_dim     = dim
        self.concat_dim        = concat_dim
        self.layer_norm        = norm_layer(dim)


        self.post_extract_proj = nn.Linear(self.concat_dim, self.embedding_dim) if self.concat_dim is not None else nn.Identity()


        self.conv_audio = nn.Sequential(
            nn.Conv1d(dim, dim, 1, 1, 0),
            nn.BatchNorm1d(dim),
            nn.PReLU(),
        )


        self.conv_video = nn.Sequential(
            nn.Conv1d(dim, dim, 1, 1, 0),
            nn.BatchNorm1d(dim),
            nn.PReLU(),
        )


    def forward(self, x_a: torch.Tensor, x_v: torch.Tensor, x: torch.Tensor): # x_a, x_v的shape均为[B, N, C] x的shape为[B, N, 2C] (只有第一次为2C)
        _, _, C = x.shape


        if self.concat_dim == C: # 均为1024
            x = self.post_extract_proj(x) # [B, N, 2C] -> [B, N, C]


        q, k, v        = x, x_a.transpose(1, 2).contiguous(), x_a  # q=[B, N, C], k=[B, C, N], v=[B, N, C]

        attn_map       = torch.softmax(torch.matmul(q, k) / math.sqrt(self.embedding_dim), dim=-1)

        residual_audio = self.conv_audio(torch.matmul(attn_map, v).transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        # video residual
        q, k, v        = x, x_v.transpose(1, 2).contiguous(), x_v  # q=[B, N, C], k=[B, C, N], v=[B, N, C]

        attn_map       = torch.softmax(torch.matmul(q, k) / math.sqrt(self.embedding_dim), dim=-1)

        residual_video = self.conv_video(torch.matmul(attn_map, v).transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        x = x + residual_audio + residual_video # NOTE: output=[B, N, C]

        x = self.layer_norm(x)

        return x # 经过强化的原始张量


class CSBlock_new_no_Mask(nn.Module):
    def __init__(self, dim, context_dim, num_heads, num_cross_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 attn_head_dim=None, cross_attn_head_dim=None,
                 ):
        super().__init__()


        self.cross_attn    = GeneralAttention(dim=dim, context_dim=context_dim, num_heads=num_cross_heads, qkv_bias=qkv_bias, 
                                           qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=cross_attn_head_dim) # 跨注意力
        
        self.cross_norm1   = norm_layer(dim)         # Q  related
        self.cross_norm2   = norm_layer(context_dim) # KV related

        # for audio
        self.cross_attn_a  = GeneralAttention(dim=dim, context_dim=context_dim, num_heads=num_cross_heads, qkv_bias=qkv_bias, 
                                           qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=cross_attn_head_dim) # 跨注意力
        
        self.cross_norm1_a = norm_layer(dim)         # Q  related
        self.cross_norm2_a = norm_layer(context_dim) # KV related


        self.norm1     = norm_layer(dim)
        self.attn      = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, 
                                    proj_drop=drop, attn_head_dim=attn_head_dim) # 自注意力


        self.norm_new  = norm_layer(dim)
        self.attn_new  = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, 
                                    proj_drop=drop, attn_head_dim=attn_head_dim) # 自注意力
        

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        mlp_hidden_dim      = int(dim * mlp_ratio)

        self.norm_vsa       = norm_layer(dim)
        self.mlp_vsa        = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm_asa       = norm_layer(dim)
        self.mlp_asa        = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm_vcma      = norm_layer(dim)
        self.mlp_vcma       = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm_acma      = norm_layer(dim)
        self.mlp_acma       = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


        if init_values > 0:
            self.gamma_0 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_0, self.gamma_1, self.gamma_2 = None, None, None


        self.norm3          = norm_layer(dim)
        self.norm3_1        = norm_layer(dim)


        self.lv1 = nn.Linear(dim * 2, dim)
        self.lv2 = nn.Linear(dim * 2, dim)

        self.la1 = nn.Linear(dim * 2, dim)
        self.la2 = nn.Linear(dim * 2, dim)
    
    def forward(self, video, audio):
        if self.gamma_1 is None: # NOTE: by default

            #------------------------#
            # 视频和音频的自注意力层
            #------------------------#
            video_sa = video + self.drop_path(self.attn(self.norm1(video)))        # 视频自注意力
            audio_sa = audio + self.drop_path(self.attn_new(self.norm_new(audio))) # 音频自注意力

            v_sa     = video_sa + self.drop_path(self.mlp_vsa(self.norm_vsa(video_sa))) # 视频自注意力前向传播层
            a_sa     = audio_sa + self.drop_path(self.mlp_asa(self.norm_asa(audio_sa))) # 音频自注意力前向传播层


            v_cma    = video + self.drop_path(self.cross_attn(self.cross_norm1(video), context=self.cross_norm2(audio)))       # 视频跨注意力
            a_cma    = audio + self.drop_path(self.cross_attn_a(self.cross_norm1_a(audio), context=self.cross_norm2_a(video))) # 音频跨注意力

            v_cma    = v_cma + self.drop_path(self.mlp_vcma(self.norm_vcma(v_cma))) # 视频跨注意力前向传播层
            a_cma    = a_cma + self.drop_path(self.mlp_acma(self.norm_acma(a_cma))) # 音频跨注意力前向传播层


            v_sq  = torch.cat((v_sa, v_cma), -1)
            v_e1  = torch.sigmoid(self.lv1(v_sq))
            v_e2  = torch.sigmoid(self.lv2(v_sq))
            v_out = torch.mul(v_e1, v_sa) + torch.mul(v_e2, v_cma)
            v_out = self.norm3(v_out)   # new added


            a_sq  = torch.cat((a_sa, a_cma), -1)
            a_e1  = torch.sigmoid(self.la1(a_sq))
            a_e2  = torch.sigmoid(self.la2(a_sq))
            a_out = torch.mul(a_e1, a_sa) + torch.mul(a_e2, a_cma)
            a_out = self.norm3_1(a_out) # new added

        else:
            raise NotImplementedError('The gamma in CSBlock_new_no_Mask is not NONE!!')

        return v_out, a_out


class VisionTransformerEncoderForFusion_new_no_Mask_IR(nn.Module):
    def __init__(self, embed_dim=768, embed_dim_audio=768, depth=3,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,):
        super().__init__()

        dpr         = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule # 为下面的融合模块去服务 # dpr中的每个元素为0到drop_path_rate的等间隔值


        self.blocks = nn.ModuleList([
            CSBlock_new_no_Mask(dim=embed_dim, context_dim=embed_dim_audio, num_heads=num_heads,
                    num_cross_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values,
                    )
            for i in range(depth)])


        self.Iterative_Refinement_Module = IterativeRefinement(dim=embed_dim, concat_dim=2 * embed_dim)


        self.cross_attn                  = GeneralAttention(dim=embed_dim, context_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                           qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate, attn_head_dim=None) # 跨注意力


        self.cross_norm_1     = norm_layer(embed_dim)       
        self.cross_norm_2     = norm_layer(embed_dim)
        self.cross_norm_3     = norm_layer(embed_dim_audio)

        self.norm       = norm_layer(embed_dim) # 视频norm层
        self.norm_audio = norm_layer(embed_dim_audio) # do not share norm layer # 音频norm层


        self.vadaptiveinteraction  = Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate, attn_head_dim=None)
        self.aadaptiveinteraction  = Attention(embed_dim_audio, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate, attn_head_dim=None)

        self.vselectfusion         = nn.Linear(embed_dim, 1)
        self.aselectfusion         = nn.Linear(embed_dim_audio, 1)


        mlp_hidden_dim        = int(embed_dim * mlp_ratio)
        mlp_hidden_dim_audio  = int(embed_dim_audio * mlp_ratio)

        # video
        self.norm_att_v       = norm_layer(embed_dim)
        self.mlp_att_v        = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop_rate)             # act_layer不需要加()

        # audio
        self.norm_att_a       = norm_layer(embed_dim_audio)
        self.mlp_att_a        = Mlp(in_features=embed_dim_audio, hidden_features=mlp_hidden_dim_audio, act_layer=nn.GELU, drop=drop_rate) # act_layer不需要加()

        self.norm_att_v_IR    = norm_layer(embed_dim)
        self.norm_att_a_IR    = norm_layer(embed_dim_audio)

        self.mlp_att_video_IR = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop_rate)             # act_layer不需要加()
        self.mlp_att_audio_IR = Mlp(in_features=embed_dim_audio, hidden_features=mlp_hidden_dim_audio, act_layer=nn.GELU, drop=drop_rate) # act_layer不需要加()


        drop_path_rate_1      = drop_path_rate * 0.5
        print(f'NOTE==> The drop_path_rate in fusion encoder during fine-tuning is {drop_path_rate_1}')

        self.drop_path        = DropPath(drop_path_rate_1) if drop_path_rate_1 > 0. else nn.Identity()


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


    def forward(self, x, x_audio, x_overall=None):


        B, N, C       = x.shape
        B_a, N_a, C_a = x_audio.shape

        assert B == B_a, f"batch sizes of video and audio do not match: {B} != {B_a}"
        assert N == N_a, f"tokens of video and audio do not match: {N} != {N_a}"
        assert C == C_a, f"channels of video and audio do not match: {C} != {C_a}"

        video_stage_list = []
        audio_stage_list = []
        for blk in self.blocks:
            x, x_audio   = blk(video=x, audio=x_audio) # [20, 8, 512]
            x_overall    = self.Iterative_Refinement_Module(x_audio, x, x_overall) 

            video_stage_list.append(x)
            audio_stage_list.append(x_audio)
        
        # print(f'The length of video_stage_list is {len(video_stage_list)}') # 2
        # print(f'The length of audio_stage_list is {len(audio_stage_list)}') # 2

        v_stage = torch.stack(video_stage_list, dim=2) # stage_num * [B, N, C] -> [B, N, stage_num, C]
        a_stage = torch.stack(audio_stage_list, dim=2) # stage_num * [B, N, C] -> [B, N, stage_num, C]

        # print('After blks and stacking')
        # print(f'video shape {v_stage.shape}') # [20, 8, 2, 512]
        # print(f'audio shape {a_stage.shape}') # [20, 8, 2, 512]

        v_stage = v_stage.view(-1, v_stage.size(2), v_stage.size(3)) # [B * N, stage_num, C]
        a_stage = a_stage.view(-1, a_stage.size(2), a_stage.size(3)) # [B * N, stage_num, C]


        v_interact    = v_stage + self.drop_path(self.vadaptiveinteraction(self.norm(v_stage)))       # [B * N, stage_num, C]
        a_interact    = a_stage + self.drop_path(self.aadaptiveinteraction(self.norm_audio(a_stage))) # [B * N, stage_num, C]

        v_interact    = v_interact + self.drop_path(self.mlp_att_v(self.norm_att_v(v_interact))) # [B * N, stage_num, C]
        a_interact    = a_interact + self.drop_path(self.mlp_att_a(self.norm_att_a(a_interact))) # [B * N, stage_num, C]


        v_weight      = torch.sigmoid(self.vselectfusion(v_interact)) # [B * N, stage_num, 1]
        a_weight      = torch.sigmoid(self.aselectfusion(a_interact)) # [B * N, stage_num, 1]

        v_interact    = v_interact.permute(0, 2, 1).contiguous() # [B * N, C, stage_num]
        a_interact    = a_interact.permute(0, 2, 1).contiguous() # [B * N, C, stage_num]

        v_out         = torch.bmm(v_interact, v_weight).view(-1, N, C)     # [B, N, C]
        a_out         = torch.bmm(a_interact, a_weight).view(-1, N_a, C_a) # [B, N, C]

        v_out   = v_out + self.drop_path(self.cross_attn(self.cross_norm_1(v_out), context=self.cross_norm_2(x_overall))) # cross-att
        a_out   = a_out + self.drop_path(self.cross_attn(self.cross_norm_3(a_out), context=self.cross_norm_2(x_overall))) # cross-att

        v_out   = v_out + self.drop_path(self.mlp_att_video_IR(self.norm_att_v_IR(v_out))) # FFN # [B, N, C]
        a_out   = a_out + self.drop_path(self.mlp_att_audio_IR(self.norm_att_a_IR(a_out))) # FFN # [B, N, C]

        return v_out, a_out, x_overall # 三个张量的shape均为[B, N, C]=[B, 8, 512] 相当于均为一致的 [2024.7.17 16.29]


class PretrainVisionTransformerEncoderForFusion_new_no_Mask_IR(nn.Module):
    """ 
    For Audio-Visual Fusion with Iterative Refinement during Pre-Train.
    """
    def __init__(self, embed_dim=768, embed_dim_audio=768, depth=3,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 modal_param_sharing=False,):
        super().__init__()

        self.modal_param_sharing = modal_param_sharing # NOTE: no use [2024.7.15 21.53]

        dpr                      = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule


        self.blocks = nn.ModuleList([
            CSBlock_new_no_Mask(dim=embed_dim, context_dim=embed_dim_audio, num_heads=num_heads,
                    num_cross_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values,
                    )
            for i in range(depth)])
        
        self.norm       = norm_layer(embed_dim)       # for video
        self.norm_audio = norm_layer(embed_dim_audio) # for audio


        self.Iterative_Refinement_Module = IterativeRefinement(dim=embed_dim, concat_dim=2 * embed_dim)


        self.cross_attn       = GeneralAttention(dim=embed_dim, context_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                           qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate, attn_head_dim=None) # 跨注意力


        self.cross_norm_1     = norm_layer(embed_dim)       # 面向视频张量
        self.cross_norm_2     = norm_layer(embed_dim)       # 面向原始张量, shape为[20, 1, 512]
        self.cross_norm_3     = norm_layer(embed_dim_audio) # 面向音频张量


        self.vadaptiveinteraction  = Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate, attn_head_dim=None)
        self.aadaptiveinteraction  = Attention(embed_dim_audio, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=drop_rate, attn_head_dim=None)

        self.vselectfusion         = nn.Linear(embed_dim, 1)
        self.aselectfusion         = nn.Linear(embed_dim_audio, 1)


        mlp_hidden_dim        = int(embed_dim * mlp_ratio)
        mlp_hidden_dim_audio  = int(embed_dim_audio * mlp_ratio)

        # video
        self.norm_att_v       = norm_layer(embed_dim)
        self.mlp_att_v        = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop_rate)             # act_layer不需要加()

        # audio
        self.norm_att_a       = norm_layer(embed_dim_audio)
        self.mlp_att_a        = Mlp(in_features=embed_dim_audio, hidden_features=mlp_hidden_dim_audio, act_layer=nn.GELU, drop=drop_rate) # act_layer不需要加()


        self.norm_att_v_IR    = norm_layer(embed_dim)
        self.norm_att_a_IR    = norm_layer(embed_dim_audio)

        self.mlp_att_video_IR = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop_rate)             # act_layer不需要加()
        self.mlp_att_audio_IR = Mlp(in_features=embed_dim_audio, hidden_features=mlp_hidden_dim_audio, act_layer=nn.GELU, drop=drop_rate) # act_layer不需要加()


        drop_path_rate_1      = drop_path_rate * 0.5
        print(f'NOTE==> The drop_path_rate in fusion encoder during pre-training is {drop_path_rate_1}')

        self.drop_path        = DropPath(drop_path_rate_1) if drop_path_rate_1 > 0. else nn.Identity()


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


    def forward(self, x, x_audio, x_overall=None):

        B, N, C       = x.shape
        B_a, N_a, C_a = x_audio.shape

        assert B == B_a, f"During Pre-training, batch sizes of video and audio do not match: {B} != {B_a}"
        assert C == C_a, f"During Pre-training, channels of video and audio do not match: {C} != {C_a}"


        video_stage_list = []
        audio_stage_list = []
        for blk in self.blocks:
            x, x_audio = blk(video=x, audio=x_audio)
            x_overall  = self.Iterative_Refinement_Module(x_audio, x, x_overall)

            video_stage_list.append(x)
            audio_stage_list.append(x_audio)



        v_stage = torch.stack(video_stage_list, dim=2) # stage_num * [B, N, C] -> [B, N, stage_num, C]
        a_stage = torch.stack(audio_stage_list, dim=2) # stage_num * [B, N, C] -> [B, N, stage_num, C]



        v_stage = v_stage.view(-1, v_stage.size(2), v_stage.size(3)) # [B * N, stage_num, C]
        a_stage = a_stage.view(-1, a_stage.size(2), a_stage.size(3)) # [B * N, stage_num, C]


        v_interact    = v_stage + self.drop_path(self.vadaptiveinteraction(self.norm(v_stage)))       # [B * N, stage_num, C]
        a_interact    = a_stage + self.drop_path(self.aadaptiveinteraction(self.norm_audio(a_stage))) # [B * N, stage_num, C]

        v_interact    = v_interact + self.drop_path(self.mlp_att_v(self.norm_att_v(v_interact)))      # [B * N, stage_num, C]
        a_interact    = a_interact + self.drop_path(self.mlp_att_a(self.norm_att_a(a_interact)))      # [B * N, stage_num, C]

        v_weight      = torch.sigmoid(self.vselectfusion(v_interact)) # [B * N, stage_num, 1]
        a_weight      = torch.sigmoid(self.aselectfusion(a_interact)) # [B * N, stage_num, 1]

        v_interact    = v_interact.permute(0, 2, 1).contiguous() # [B * N, C, stage_num]
        a_interact    = a_interact.permute(0, 2, 1).contiguous() # [B * N, C, stage_num]

        v_out         = torch.bmm(v_interact, v_weight).view(-1, N, C)     # [B, N, C]
        a_out         = torch.bmm(a_interact, a_weight).view(-1, N_a, C_a) # [B, N, C]


        v_out   = v_out + self.drop_path(self.cross_attn(self.cross_norm_1(v_out), context=self.cross_norm_2(x_overall)))
        a_out   = a_out + self.drop_path(self.cross_attn(self.cross_norm_3(a_out), context=self.cross_norm_2(x_overall)))

        v_out   = v_out + self.drop_path(self.mlp_att_video_IR(self.norm_att_v_IR(v_out)))
        a_out   = a_out + self.drop_path(self.mlp_att_audio_IR(self.norm_att_a_IR(a_out)))

        return v_out, a_out, video_stage_list, audio_stage_list