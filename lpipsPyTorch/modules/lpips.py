import torch
import torch.nn as nn

from .networks import LinLayers, get_network
from .utils import get_state_dict


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """

    def __init__(self, net_type: str = "alex", version: str = "0.1"):

        assert version in ["0.1"], "v0.1 is only supported now"

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        # Lower memory usage: process one layer at a time, squared diff
        # in place in fx's memory, freeing each layer after use
        res = []
        for l in self.lin:
            fx, fy = feat_x.pop(0), feat_y.pop(0)
            res.append(l(fx.sub_(fy).pow_(2)).mean((2, 3), True))
            del fx, fy

        return torch.sum(torch.cat(res, 0), 0, True)
