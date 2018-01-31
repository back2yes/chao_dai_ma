from torch import nn


class _RoIPooling(nn.Module):
    def __init__(self):
        super(_RoIPooling, self).__init__()
