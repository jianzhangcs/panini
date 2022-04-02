import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmedit.models.backbones.rrdb_net import RRDB
from mmedit.models.builder import build_component
from mmedit.models.common import PixelShufflePack, make_layer
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


class AdaptiveFeatureSelect_maskgen(nn.Module):
    def __init__(self, outchannel=512*5, stylechannel=256 ,hard_fusion=False):
        super(AdaptiveFeatureSelect_maskgen,self).__init__()
        self.hardfusion = hard_fusion
        self.outchannel =outchannel
        self.mapping = nn.Sequential(
            nn.Linear(stylechannel, stylechannel),
            nn.LeakyReLU(0.1, True),
            nn.Linear(stylechannel, outchannel*2),
            )
    def forward(self, style):
        mask = self.mapping(style)
        mask = torch.unsqueeze(mask, 1).view(style.shape[0], 5, 2, 512, 1, 1)
        mask = torch.softmax(mask, 2)
        return mask[:, :, 0, :, :, :].squeeze(1), mask[:, :, 1, :, :, :].squeeze(1)


class AdaptiveFeatureSelect(nn.Module):
    def __init__(self, outchannel=512, stylechannel=256 ,hard_fusion=False):
        super(AdaptiveFeatureSelect,self).__init__()
        self.hardfusion = hard_fusion
        self.outchannel =outchannel
        self.mapping = nn.Sequential(
            nn.Linear(stylechannel, stylechannel),
            nn.LeakyReLU(0.1, True),
            nn.Linear(stylechannel, outchannel*2),
            )
    def forward(self, fea1, fea2, style):
        mask = self.mapping(style)
        mask = torch.unsqueeze(mask, 1).view(-1, 2, self.outchannel, 1, 1)
        mask = torch.softmax(mask, 1)
        fea1 = fea1 * mask[:, 0, :, :, :].squeeze(1)
        fea2 = fea2 * mask[:, 1, :, :, :].squeeze(1)
        return fea1 + fea2    


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0 \
            , dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size \
                , stride=stride, padding=padding, dilation=dilation \
                , groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01 \
                , affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16 \
            , pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, fea_preprocess):

        x = fea_preprocess(x)

        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = self.avgpool(x)
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = self.maxpool(x)
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = self.sigmoid(channel_att_sum).unsqueeze(2) \
                .unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1) \
                .unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1 \
                , padding=(kernel_size-1) // 2, relu=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, fea_preprocess):
        x = fea_preprocess(x)
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16 \
            , pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.fea_preprocess = nn.Conv2d(gate_channels, gate_channels, 3, 1, 1)
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x, self.fea_preprocess)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out, self.fea_preprocess)
        return x_out
    
class SpatialGate_interpolation(nn.Module):
    def __init__(self):
        super(SpatialGate_interpolation, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1 \
                , padding=(kernel_size-1) // 2, relu=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, y, fea_preprocess):
        x = fea_preprocess(x)
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        mask = self.sigmoid(x_out) # broadcasting
        #print('mask:',mask.shape)
        #print('x:',x.shape)
        return x*mask + y*(1-mask)
    
class CBAM_spatial_interpolation(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16 \
            , pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM_spatial_interpolation, self).__init__()
        self.fea_preprocess = nn.Conv2d(gate_channels, gate_channels, 3, 1, 1)
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate_interpolation = SpatialGate_interpolation()
    def forward(self, x, y):
        x_out = self.SpatialGate_interpolation(x, y, self.fea_preprocess)
        return x_out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        #print('x.shape:',x.shape)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        #out = self.mlp(fea)
        #print('fea.shape:',fea.shape)
        return fea

class RRDBFeatureExtractor(nn.Module):
    """Feature extractor composed of Residual-in-Residual Dense Blocks (RRDBs).

    It is equivalent to ESRGAN with the upsampling module removed.

    Args:
        in_channels (int): Channel number of inputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Default: 23
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 in_channels=3,
                 mid_channels=64,
                 num_blocks=23,
                 growth_channels=32):

        super().__init__()

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(
            RRDB,
            num_blocks,
            mid_channels=mid_channels,
            growth_channels=growth_channels)
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        feat = self.conv_first(x)
        return feat + self.conv_body(self.body(feat))

@BACKBONES.register_module()
class Panini_MFR(nn.Module):

    def __init__(self,
                in_size,
                out_size,
                img_channels=3,
                #rrdb_channels=64,
                #num_rrdbs=23,
                rrdb_channels=8,
                num_rrdbs=2,
                style_channels=512,
                num_mlps=8,
                channel_multiplier=2,
                blur_kernel=[1, 3, 3, 1],
                lr_mlp=0.01,
                default_style_mode='mix',
                eval_style_mode='single',
                mix_prob=0.9,
                pretrained=None,
                bgr2rgb=False):

        super().__init__()

        # input size must be strictly smaller than output size
        if in_size >= out_size:
            raise ValueError('in_size must be smaller than out_size, but got '
                            f'{in_size} and {out_size}.')

        # latent bank (StyleGANv2), with weights being fixed
        self.generator = build_component(
            dict(
                type='StyleGANv2Generator',
                out_size=out_size,
                style_channels=style_channels,
                num_mlps=num_mlps,
                channel_multiplier=channel_multiplier,
                blur_kernel=blur_kernel,
                lr_mlp=lr_mlp,
                default_style_mode=default_style_mode,
                eval_style_mode=eval_style_mode,
                mix_prob=mix_prob,
                pretrained=pretrained,
                bgr2rgb=bgr2rgb))
        self.generator.requires_grad_(True)

        self.in_size = in_size
        self.style_channels = style_channels
        channels = self.generator.channels

        #resize img for deg_encoder
        self.resize = nn.AdaptiveAvgPool2d((256, 256))

        # deg_encoder
        self.deg_encoder = Encoder()
        # load statedict for deg_encoder
        with torch.no_grad():
            model_dict = self.deg_encoder.state_dict()
            pretrained_dict = torch.load('checkpoint/moco_checkpoint_0199.pth.tar', map_location='cpu')
            keys = []
            for k, v in pretrained_dict['state_dict'].items():
                keys.append(k)
                #print(k)
            i = 2
            for k, v in model_dict.items():
                #print(k)
                if v.size() == pretrained_dict['state_dict'][keys[i]].size():
                    model_dict[k] = pretrained_dict['state_dict'][keys[i]]
                    i = i + 1
                    #print(k)
                else:
                    print("miss!")
            print("load stat_edict for deg_encoder")
            self.deg_encoder.load_state_dict(model_dict)
            self.deg_encoder.requires_grad_(True)

        # encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    img_channels, rrdb_channels, num_blocks=num_rrdbs),
                CBAM(rrdb_channels, reduction_ratio=2),
                nn.Conv2d(
                    rrdb_channels, 32, 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),#res 512
                nn.Conv2d(32, 64, 3, 2, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),#res 256
                nn.Conv2d(64, 128, 3, 2, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),#res 128
                nn.Conv2d(128, 512, 3, 2, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),#res 64
            )
        )
        for i in range(4):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(512, 512, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),#res 32 ~ res 4
                )
            )
        self.encoder.append(
            nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),#res w
                nn.Flatten(),
                nn.Linear(16 * 256, 18*style_channels),
            )
        )

        # additional modules for StyleGANv2
        self.AFS_maskGen = AdaptiveFeatureSelect_maskgen()
        self.fusion_out = nn.ModuleList()
        for res in [64, 32, 16, 8, 4]:
            num_channels = channels[res]
            self.fusion_out.append(
                nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=True)
                )


    def forward(self, lq):
        """Forward function.

        Args:
            lq (Tensor): Input LR image with shape (n, c, h, w).

        Returns:
            Tensor: Output HR image.
        """

        h, w = lq.shape[2:]
        if h != self.in_size or w != self.in_size:
            raise AssertionError(
                f'Spatial resolution must equal in_size ({self.in_size}).'
                f' Got ({h}, {w}).')

        # deg_encoder
        img_deg = self.resize(lq)
        deg_style = self.deg_encoder(img_deg)


        #AFS_maskgen, shape:mask[batch, layer, channel, 1, 1]
        afs_mask1, afs_mask2 = self.AFS_maskGen(deg_style)
        #print('afs_mask.shape:',afs_mask1.shape)

        # encoder
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]

        latent = encoder_features[0].view(lq.size(0), -1, self.style_channels)
        #print('latent.shape:',latent.shape)
        encoder_features = encoder_features[1:]

        # generator
        injected_noise = [
            getattr(self.generator, f'injected_noise_{i}')
            for i in range(self.generator.num_injected_noises)
        ]

        # 4x4 stage
        out = self.generator.constant_input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.generator.to_rgb1(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher res
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.generator.convs[::2], self.generator.convs[1::2],
                injected_noise[1::2], injected_noise[2::2],
                self.generator.to_rgbs):

            # feature fusion for res4 - res64
            if out.size(2) <= 64:
                fusion_index = (_index - 1) // 2
                feat = encoder_features[fusion_index]
                #print('feat.shape:',feat.shape)

                # mask[batch, layer, channel, 1, 1]
                feat = self.fusion_out[fusion_index](feat)
                out = afs_mask1[:, fusion_index, :, :, :].squeeze(1) *out + afs_mask2[:, fusion_index, :, :, :].squeeze(1) *feat
                

            # original StyleGAN operations
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index+1], noise=noise2)
            skip = to_rgb(out, latent[:, _index+2], skip)

            _index += 2

        return skip

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
                            

            
@BACKBONES.register_module()
class Panini_SR(nn.Module):

    def __init__(self,
                in_size,
                out_size,
                img_channels=3,
                rrdb_channels=64,
                num_rrdbs=23,
                #rrdb_channels=8,
                #num_rrdbs=2,
                style_channels=512,
                num_mlps=8,
                channel_multiplier=2,
                blur_kernel=[1, 3, 3, 1],
                lr_mlp=0.01,
                default_style_mode='mix',
                eval_style_mode='single',
                mix_prob=0.9,
                pretrained=None,
                bgr2rgb=False):

        super().__init__()

        # input size must be strictly smaller than output size
        if in_size >= out_size:
            raise ValueError('in_size must be smaller than out_size, but got '
                            f'{in_size} and {out_size}.')

        # latent bank (StyleGANv2), with weights being fixed
        self.generator = build_component(
            dict(
                type='StyleGANv2Generator',
                out_size=out_size,
                style_channels=style_channels,
                num_mlps=num_mlps,
                channel_multiplier=channel_multiplier,
                blur_kernel=blur_kernel,
                lr_mlp=lr_mlp,
                default_style_mode=default_style_mode,
                eval_style_mode=eval_style_mode,
                mix_prob=mix_prob,
                pretrained=pretrained,
                bgr2rgb=bgr2rgb))
        self.generator.requires_grad_(True)

        self.in_size = in_size
        self.style_channels = style_channels
        channels = self.generator.channels

        self.choice_vector = nn.Parameter(torch.randn(1, 256))
        
        # encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                RRDBFeatureExtractor(
                    img_channels, rrdb_channels, num_blocks=num_rrdbs),
                CBAM(rrdb_channels, reduction_ratio=8),
                nn.Conv2d(
                    rrdb_channels, 512, 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),#res 64
            )
        )
        for i in range(4):
            print(i)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(512, 512, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),#res 32 ~ res 4
                )
            )
        self.encoder.append(
            nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),#res w
                nn.Flatten(),
                nn.Linear(16 * 256, 18*style_channels),
            )
        )

        # additional modules for StyleGANv2
        self.AFS_maskGen = AdaptiveFeatureSelect_maskgen()
        self.fusion_out = nn.ModuleList()
        for res in [64, 32, 16, 8, 4]:
            num_channels = channels[res]
            self.fusion_out.append(
                nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=True)
                )


    def forward(self, lq):
        """Forward function.

        Args:
            lq (Tensor): Input LR image with shape (n, c, h, w).

        Returns:
            Tensor: Output HR image.
        """

        h, w = lq.shape[2:]
        if h != self.in_size or w != self.in_size:
            raise AssertionError(
                f'Spatial resolution must equal in_size ({self.in_size}).'
                f' Got ({h}, {w}).')
            
        deg_style = self.choice_vector.repeat(lq.shape[0], 1, 1)

        #AFS_maskgen, shape:mask[batch, layer, channel, 1, 1]
        afs_mask1, afs_mask2 = self.AFS_maskGen(deg_style)
        #print('afs_mask.shape:',afs_mask1.shape)

        # encoder
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]

        latent = encoder_features[0].view(lq.size(0), -1, self.style_channels)
        #print('latent.shape:',latent.shape)
        encoder_features = encoder_features[1:]

        # generator
        injected_noise = [
            getattr(self.generator, f'injected_noise_{i}')
            for i in range(self.generator.num_injected_noises)
        ]

        # 4x4 stage
        out = self.generator.constant_input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.generator.to_rgb1(out, latent[:, 1])

        _index = 1

        # 8x8 ---> higher res
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.generator.convs[::2], self.generator.convs[1::2],
                injected_noise[1::2], injected_noise[2::2],
                self.generator.to_rgbs):

            # feature fusion for res4 - res64
            if out.size(2) <= 64:
                fusion_index = (_index - 1) // 2
                feat = encoder_features[fusion_index]
                #print('feat.shape:',feat.shape)

                # mask[batch, layer, channel, 1, 1]
                feat = self.fusion_out[fusion_index](feat)
                out = afs_mask1[:, fusion_index, :, :, :].squeeze(1) *out + afs_mask2[:, fusion_index, :, :, :].squeeze(1) *feat
                

            # original StyleGAN operations
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index+1], noise=noise2)
            skip = to_rgb(out, latent[:, _index+2], skip)

            _index += 2

        return skip

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

