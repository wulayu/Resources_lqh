import pprint
import torch
from torch import nn
import torch.nn.functional as F

from models.MobileNetV2 import MobileNetV2, InvertedResidual, conv_1x1_bn

default_cnn_cfg = {
    'widen_factor': 1., 'pretrained': True, 'input_channel': 32,
    'last_channel': 1280}

class MobileSal(nn.Module):
    def __init__(self,
                 cnn_cfg=None,
                 cnn_out_chn=256,
                 drop_probs=(0.0, 0.6, 0.6),
                 bn_momentum=0.01,
                 ):
        super().__init__()

        this_cnn_cfg = default_cnn_cfg.copy()
        this_cnn_cfg.update(cnn_cfg or {})
        self.cnn_cfg = this_cnn_cfg
        self.drop_probs = drop_probs
        self.bn_momentum = bn_momentum

        # Initialize backbone CNN
        self.cnn = MobileNetV2(**self.cnn_cfg)

        post_cnn = [InvertedResidual(self.cnn.out_channels,cnn_out_chn, 1, 1, bn_momentum=bn_momentum,)]

        if self.drop_probs[0] > 0:
            post_cnn.append(nn.Dropout2d(self.drop_probs[0], inplace=False))
        self.post_cnn = nn.Sequential(*post_cnn)

        self.upsampling_1 = self.upsampling(2)

        channels_2x = 128
        self.skip_2x = self.make_skip_connection(
            self.cnn.feat_2x_channels, channels_2x, 2, self.drop_probs[1])

        self.upsampling_2 = nn.Sequential(
            InvertedResidual(cnn_out_chn + channels_2x, channels_2x, 1, 2),
            self.upsampling(2),
        )

        channels_4x = 64
        self.skip_4x = self.make_skip_connection(
            self.cnn.feat_4x_channels, channels_4x, 2, self.drop_probs[2])

        self.post_upsampling_2= nn.Sequential(
            InvertedResidual(channels_2x + channels_4x, channels_4x, 1, 2,),
        )

        self.final_conv = conv_1x1_bn(channels_4x, 1)

    def upsampling(self, factor):
        return nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)

    def make_skip_connection(self, input_channels, output_channels, expand_ratio, p,
                       inplace=False):
        hidden_channels = round(input_channels * expand_ratio)
        return nn.Sequential(
            self.conv_1x1_bn(input_channels, hidden_channels),
            nn.Dropout2d(p, inplace=inplace),
            nn.Conv2d(hidden_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
        )

    def conv_1x1_bn(self, inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x, target_size=None,):
        if target_size is None:
            target_size = x.shape[-2:]

        im_feat_1x, im_feat_2x, im_feat_4x = self.cnn(x)
        im_feat_2x = self.skip_2x(im_feat_2x)
        im_feat_4x = self.skip_4x(im_feat_4x)
        im_feat_1x = self.post_cnn(im_feat_1x)

        im_feat = self.upsampling_1(im_feat_1x)
        im_feat = torch.cat((im_feat, im_feat_2x), dim=1)
        im_feat = self.upsampling_2(im_feat)
        im_feat = torch.cat((im_feat, im_feat_4x), dim=1)
        im_feat = self.post_upsampling_2(im_feat)
        im_feat = self.final_conv(im_feat)
        im_feat = F.interpolate(
            im_feat, size=x.shape[-2:], mode='nearest')

        im_feat = F.interpolate(
            im_feat, size=target_size, mode='bilinear', align_corners=False)
        
        return im_feat
    

if __name__ == "__main__":
    model = MobileSal()
    data = torch.randn(1, 3, 384, 256)
    out = model(data)
