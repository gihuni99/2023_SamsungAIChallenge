import importlib
import torch.nn as nn
from torch.nn import functional as F
from .decoder import *
from .resnet import *


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        self._sync_bn = True
        self._num_classes = 13

        self.encoder = resnet101(pretrained=True, 
                                        pretrain_model_url='./utils/pretrained/resnet101.pth',
                                        zero_init_residual=True,
                                        multi_grid=True,
                                        replace_stride_with_dilation=[False, False, True],
                                        sync_bn=True
                                        )
        self.decoder = dec_deeplabv3_plus(in_planes=self.encoder.get_outplanes(),
                                           num_classes=13,
                                          inner_planes=256,
                                          sync_bn=True,
                                          dilations=(6, 12, 18),
                                          low_conv_planes=48
                                        )


    def forward(self, x, flag_use_fdrop=True): #dropout설정
        h, w = x.shape[-2:]
        if flag_use_fdrop:
            f1, f2, feat1, feat2 = self.encoder(x)
            f1 = nn.Dropout2d(0.5)(f1)
            feat2 = nn.Dropout2d(0.5)(feat2)
            outs = self.decoder([f1, f2, feat1, feat2])
        else:
            feat = self.encoder(x)
            outs = self.decoder(feat)
        outs = F.interpolate(outs, (h, w), mode="bilinear", align_corners=True)
            
        return outs
