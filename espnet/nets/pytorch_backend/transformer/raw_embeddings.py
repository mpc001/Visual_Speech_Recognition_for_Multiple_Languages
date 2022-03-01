import torch
import logging

from espnet.nets.pytorch_backend.backbones.conv3d_extractor  import Conv3dResNet
from espnet.nets.pytorch_backend.backbones.conv1d_extractor  import Conv1dResNet


class VideoEmbedding(torch.nn.Module):
    """Video Embedding

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, idim, odim, dropout_rate, pos_enc_class, backbone_type="resnet", relu_type="prelu"):
        super(VideoEmbedding, self).__init__()
        self.trunk = Conv3dResNet(
            backbone_type=backbone_type,
            relu_type=relu_type
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            pos_enc_class,
        )

    def forward(self, x, x_mask, extract_feats=None):
        """video embedding for x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :param str extract_features: the position for feature extraction
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x_resnet, x_mask = self.trunk(x, x_mask)
        x = self.out(x_resnet)
        if extract_feats:
            return x, x_mask, x_resnet
        else:
            return x, x_mask


class AudioEmbedding(torch.nn.Module):
    """Audio Embedding

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """

    def __init__(self, idim, odim, dropout_rate, pos_enc_class, relu_type="prelu", a_upsample_ratio=1):
        super(AudioEmbedding, self).__init__()
        self.trunk = Conv1dResNet(
            relu_type=relu_type,
            a_upsample_ratio=a_upsample_ratio,
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            pos_enc_class,
        )

    def forward(self, x, x_mask, extract_feats=None):
        """audio embedding for x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :param str extract_features: the position for feature extraction
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x_resnet, x_mask = self.trunk(x, x_mask)
        x = self.out(x_resnet)
        if extract_feats:                                                        
            return x, x_mask, x_resnet                                           
        else:                                                                    
            return x, x_mask
