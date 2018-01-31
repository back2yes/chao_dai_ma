from torch import nn


class _RoICrop(nn.Module):
    def __init__(self):
        super(_RoICrop, self).__init__()
