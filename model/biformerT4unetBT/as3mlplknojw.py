"""
BiFormer impl.

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg

from BMP_Net.code.model.biformerunet.bra_legacy import BiLevelRoutingAttention
from BMP_Net.code.model.biformerunet._common import Attention, AttentionLePE, DWConv



def get_pe_layer(emb_dim, pe_dim=None, name='none'):  #emb_dim: 表示嵌入维度（embedding dimension）
                                                      # pe_dim: 表示位置编码维度（positional encoding）
    if name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'PE name {name} is not surpported!')


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,  #初始化
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=True):
        super().__init__()
        qk_dim = qk_dim or dim
        # dim: 表示输入和输出特征的维度。
        # drop_path: 指定是否使用DropPath操作以及其概率。
        # layer_scale_init_value: 用于初始化层缩放的初始值。
        # num_heads: 注意力头的数量。
        # n_win: 注意力窗口的数量。
        # qk_dim: 查询和键的维度，默认为dim。
        # qk_scale: 查询和键的缩放因子。
        # kv_per_win: 每个窗口中的键值对的数量。
        # kv_downsample_ratio: 键值下采样的比率。
        # kv_downsample_kernel: 键值下采样时使用的卷积核。
        # kv_downsample_mode: 键值下采样的模式。
        # topk: 用于注意力操作的Top - K参数。
        # param_attention: 注意力机制中的参数设置。
        # param_routing: 是否使用参数化路由。
        # diff_routing: 是否使用不同的路由。
        # soft_routing: 是否使用软路由。
        # mlp_ratio: 多层感知机(MLP)中隐藏层的维度与输入维度的比例。
        # mlp_dwconv: 是否使用深度可分离卷积作为MLP的一部分。
        # side_dwconv: 用于侧边信息的深度可分离卷积的核大小。
        # before_attn_dwconv: 注意力之前的深度可分离卷积的核大小。
        # pre_norm: 是否在注意力和MLP之前应用层归一化。
        # auto_pad: 是否自动填充卷积。
        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                kv_downsample_ratio=kv_downsample_ratio,
                                                kv_downsample_kernel=kv_downsample_kernel,
                                                kv_downsample_mode=kv_downsample_mode,
                                                topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                diff_routing=diff_routing, soft_routing=soft_routing,
                                                side_dwconv=side_dwconv,
                                                auto_pad=auto_pad)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'),  # compatiability
                                      nn.Conv2d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
                                      nn.Conv2d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                      )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:     #用于初始化层缩放的初始值
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:  # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class BiFormer(nn.Module):
    def __init__(self, depth=[3, 4, 8, 3], in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, qk_scale=None, representation_size=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=True,
                 # -----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)  # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            x = self.stages[i](x)
            outs.append(x)
            if(len(outs)==4):
                outs[3]=self.norm(outs[3])
        # x = self.pre_logits(x)
        return outs

    def forward(self, x):

        outs = self.forward_features(x)
        return outs

        # features = []
        # outs = self.forward_features(x)
        # add_feature = F.interpolate(outs[0], scale_factor=2)
        # features = [add_feature] + outs
        # return features

#################### model variants #######################


model_urls = {
    "biformer_tiny_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHEOoGkgwgQzEDlM/root/content",
    "biformer_small_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHDyM-x9KWRBZ832/root/content",
    "biformer_base_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHI_XPhoadjaNxtO/root/content",
}


# https://github.com/huggingface/pytorch-image-models/blob/4b8cfa6c0a355a9b3cb2a77298b240213fb3b921/timm/models/_factory.py#L93

@register_model
def biformer_tiny(pretrained=True, pretrained_cfg=None,
                  pretrained_cfg_overlay=None, **kwargs):
    model = BiFormer(
        depth=[2, 2, 8, 2],
        embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        # -------------------------------
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        model_key = 'biformer_tiny_in1k'
        url = model_urls[model_key]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True,
                                                        file_name=f"{model_key}.pth")
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def biformer_small(pretrained=False, pretrained_cfg=None,
                   pretrained_cfg_overlay=None, **kwargs):
    model = BiFormer(
        depth=[4, 4, 18, 4],
        embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        # -------------------------------
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        model_key = 'biformer_small_in1k'
        url = model_urls[model_key]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True,
                                                        file_name=f"{model_key}.pth")
        model.load_state_dict(checkpoint["model"])

    return model


@register_model
def biformer_base(pretrained=False, pretrained_cfg=None,
                  pretrained_cfg_overlay=None, **kwargs):
    model = BiFormer(
        depth=[4, 4, 18, 4],
        embed_dim=[96, 192, 384, 768], mlp_ratios=[3, 3, 3, 3],
        # use_checkpoint_stages=[0, 1, 2, 3],
        use_checkpoint_stages=[],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[96, 192, 384, 768],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        # -------------------------------
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        model_key = 'biformer_base_in1k'
        url = model_urls[model_key]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True,
                                                        file_name=f"{model_key}.pth")
        model.load_state_dict(checkpoint["model"])

    return model


class unetUp(nn.Module):     #U-Net模型中上采样（upsampling）部分
    def __init__(self, in_size, out_size):     #输入、输出特征通道数
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)  #上采样层，通过双线性插值将输入的特征图的大小扩大两倍。
        self.relu = nn.ReLU(inplace=True)           #ReLU激活函数，inplace=True 表示直接在原地进行激活函数操作。

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)  #使用 torch.cat 在通道维度上拼接 inputs1 和上采样后的 inputs2。
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class shiftmlp(nn.Module):    #实现了一种带有平移操作的MLP（多层感知机）
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()                       #drop: Dropout 概率   shift_size: 平移操作的尺寸
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)   #深度可分离卷积层
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    #     def shift(x, dim):
    #         x = F.pad(x, "constant", 0)
    #         x = torch.chunk(x, shift_size, 1)
    #         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
    #         x = torch.cat(x, 1)
    #         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape    #B 是批大小，N 是序列长度，C 是通道数。

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)  #零填充，填充的尺寸由 self.pad 决定
        xs = torch.chunk(xn, self.shift_size, 1) #将填充后的张量沿着通道维度切分成 self.shift_size 份
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]#torch.roll 用于实现循环平移
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W) #将平移后的结果拼接为一个张量，然后在高度和宽度维度上进行裁剪，保留有效部分。

        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)   #重塑张量并将其转置为 (B, H * W, C)

        x = self.fc1(x_shift_r)    #输入平移后的张量到线性层 fc1

        x = self.dwconv(x, H, W)
        x = self.act(x)           #应用深度可分离卷积、激活函数和 Dropout
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()    #对通道维度进行类似的平移操作，将其应用于新的输入张量
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)     #输入平移后的张量到线性层 fc2，最后应用 Dropout
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):    #平移MLP
    def __init__(self, dim,mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()       #norm_layer: 归一化层，默认为 LayerNorm。drop_path: DropPath 操作的概率
        #dim: 输入特征的维度。 mlp_ratio: MLP 中隐藏层的维度与输入维度的比例。act_layer: 激活函数，默认为 GELU
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):   #用于初始化模型参数的方法，包括线性层的权重和偏置、归一化层的参数以及卷积层的权重和偏置。
        if isinstance(m, nn.Linear):    #线性层
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):    #归一化层
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):     #深度可分离卷积模块
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape     #维度转置，重塑为 (B, C, H, W) 的形状。
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2) #将深度可分离卷积的输出展平为 (B, N, C) 的形状，并将通道维度移回到第三维

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding      用于将图像转换为补足重叠区域的图块表示（patch embedding）stride: 卷积的步幅
                                      embed_dim: 嵌入维度
                  这个模块实现了一个基于卷积操作的图块嵌入器，用于将输入图像
                  转换为图块表示。该操作会产生具有重叠区域的图块，通过卷积的步幅和补充，确保了图块之间的信息交叠。
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):   #归一化层
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)    #接收输入张量 x，将其经过卷积层 proj 进行图块化操作
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)    #归一化
        return x, H, W    #返回归一化后的张量以及图块化后的高度和宽度信息。


class unetUpsmlp4(nn.Module):     #多层级特征融合
    def __init__(self, dim,outsize):
        super(unetUpsmlp4, self).__init__()
        self.decoder3 = nn.Conv2d(in_channels=dim, out_channels=outsize, kernel_size=3,stride=1,padding=1)
        self.dblock1 = nn.ModuleList([shiftedBlock(       #一个由 shiftedBlock 组成的列表，用于进行多层级的特征融合。
            dim=dim,mlp_ratio=1, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0.0, norm_layer=nn.LayerNorm,
            sr_ratio=8)])
        self.dnorm3 = nn.LayerNorm(dim)   #对输出进行归一化的 LayerNorm 层
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.dbn3=nn.BatchNorm2d(outsize)    #dbn3: 对最终输出进行批归一化的层

        self.convjw = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(dim),
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=9, padding=4,
                      bias=True),
            nn.BatchNorm2d(dim),
        )                                   #convjw, conv9, conv7, conv5, conv3: 分别是不同尺寸的卷积层和批归一化层的序列
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, padding=3,
                      bias=True),
            nn.BatchNorm2d(dim),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2,
                      bias=True),
            nn.BatchNorm2d(dim),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1,
                      bias=True),
            nn.BatchNorm2d(dim),
        )

    def forward(self, inputs1, inputs2):
        outputsor = torch.cat([inputs1, self.up(inputs2)], 1)   #拼接

        out1=outputsor          #smlp      对拼接后的张量进行多层级的特征融合，其中包括 shiftedBlock 模块。
        _, _, H, W = out1.shape
        out1 = out1.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out1 = blk(out1, H, W)
        out1 = self.dnorm3(out1)       #归一化
        out1 = out1.reshape(out1.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        out2=outputsor
        out2=F.relu(self.conv9(out2)+self.conv3(out2))
                                          #使用不同尺寸的卷积核进行特征融合，通过卷积和批归一化操作
        out3=outputsor
        out3 = F.relu(self.conv7(out3) + self.conv3(out3))

        out4 =outputsor
        out4 = F.relu(self.conv5(out4) + self.conv3(out4))

        out=out1+out2+out3+out4

        out=F.relu(self.dbn3(self.decoder3(out)))  #
        return out

class unetUpsmlp3(nn.Module):
    def __init__(self, dim,outsize):
        super(unetUpsmlp3, self).__init__()
        self.decoder3 = nn.Conv2d(in_channels=dim, out_channels=outsize, kernel_size=3,stride=1,padding=1)
        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=dim,mlp_ratio=1, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0.0, norm_layer=nn.LayerNorm,
            sr_ratio=8)])
        self.dnorm3 = nn.LayerNorm(dim)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.dbn3=nn.BatchNorm2d(outsize)

        self.convjw = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(dim),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, padding=3,
                      bias=True),
            nn.BatchNorm2d(dim),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2,
                      bias=True),
            nn.BatchNorm2d(dim),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1,
                      bias=True),
            nn.BatchNorm2d(dim),
        )

    def forward(self, inputs1, inputs2):
        outputsor = torch.cat([inputs1, self.up(inputs2)], 1)


        out1=outputsor         #smlp
        _, _, H, W = out1.shape
        out1 = out1.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out1 = blk(out1, H, W)
        out1 = self.dnorm3(out1)
        out1 = out1.reshape(out1.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        out2=outputsor
        out2=F.relu(self.conv7(out2)+self.conv3(out2))

        out3=outputsor
        out3 = F.relu(self.conv5(out3) + self.conv3(out3))

        out4 =outputsor
        out4 = F.relu(self.conv3(out4))

        out=out1+out2+out3+out4

        out=F.relu(self.dbn3(self.decoder3(out)))
        return out
       #不同之处在于此处的特征融合结构和卷积层设置

class unetUpsmlp2(nn.Module):
    def __init__(self, dim, outsize):
        super(unetUpsmlp2, self).__init__()
        self.decoder3 = nn.Conv2d(in_channels=dim, out_channels=outsize, kernel_size=3, stride=1, padding=1)
        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=dim, mlp_ratio=1, qkv_bias=False, qk_scale=None,
            drop=0., attn_drop=0., drop_path=0.0, norm_layer=nn.LayerNorm,
            sr_ratio=8)])
        self.dnorm3 = nn.LayerNorm(dim)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.dbn3 = nn.BatchNorm2d(outsize)

        self.convjw = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(dim),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2,
                      bias=True),
            nn.BatchNorm2d(dim),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1,
                      bias=True),
            nn.BatchNorm2d(dim),
        )

    def forward(self, inputs1, inputs2):
        outputsor = torch.cat([inputs1, self.up(inputs2)], 1)

        out1 = outputsor  # smlp
        _, _, H, W = out1.shape
        out1 = out1.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out1 = blk(out1, H, W)
        out1 = self.dnorm3(out1)
        out1 = out1.reshape(out1.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()

        out2 = F.relu(self.convjw(outputsor))   # 1 3 5

        out3 = F.relu(self.conv5(outputsor) + self.conv3(outputsor))

        out4 = F.relu(self.conv3(outputsor))

        out = out1+out2+out3+out4

        out = F.relu(self.dbn3(self.decoder3(out)))
        return out



class Unet(nn.Module):
    def __init__(self, num_classes):
        super(Unet, self).__init__()
        self.backbone = biformer_tiny()
        in_filters = [192, 320, 640, 768]
        out_filters = [64, 128, 256,512]

        self.up_concat4 = unetUpsmlp4(dim=in_filters[3],outsize=out_filters[3])
        self.up_concat3 = unetUpsmlp3(dim=in_filters[2],outsize=out_filters[2])
        self.up_concat2 = unetUpsmlp2(dim=in_filters[1],outsize=out_filters[1])

        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[1], out_filters[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[1], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv21s = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv22s = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        # 3s
        self.conv31s = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv32s = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        # 4s
        self.conv41s = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv42s = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        # 5s
        self.conv51s = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv52s = nn.Conv2d(128, 512, kernel_size=3, padding=1)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)


    def forward(self, inputs):
        [feat2, feat3, feat4,feat5] = self.backbone(inputs)[0:]   #C: 64, 128, 256, 512

        feat2 = feat2 + F.interpolate(self.conv21s(feat3), scale_factor=2) + F.interpolate(self.conv22s(feat4),
                                                                                           scale_factor=4)
        feat3 = feat3 + F.interpolate(self.conv32s(feat4), scale_factor=2) + F.interpolate(self.conv31s(feat2),
                                                                                           scale_factor=0.5)
        feat4 = feat4 + F.interpolate(self.conv42s(feat5), scale_factor=2) + F.interpolate(self.conv41s(feat3),
                                                                                           scale_factor=0.5)
        feat5 = feat5 + F.interpolate(self.conv51s(feat4), scale_factor=0.5) + F.interpolate(self.conv52s(feat3),
                                                                                             scale_factor=0.25)

        up4 = self.up_concat4(feat4, feat5)   #3个变体的mlp
        up3 = self.up_concat3(feat3, up4)     #3个变体的mlp
        up2 = self.up_concat2(feat2, up3)     #3个变体的mlp
        up1 = self.up_conv(up2)
        up1new=self.upsample(up1)
        final = self.final(up1new)
        return final

if __name__ == '__main__':
    input = torch.randn(1, 3, 256, 256)
    sft = Unet(num_classes=2)
    output=sft(input)
    print(output.shape)
