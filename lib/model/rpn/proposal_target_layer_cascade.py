from torch import nn


class _ProposalTargetLayer(nn.Module):
    def __init__(self):
        super(_ProposalTargetLayer, self).__init__()
