import torch
import torch.nn as nn

class ScaleShiftWrapper(nn.Module):
    """
    Wrap a module (Conv2d or Linear). During meta-stage, we keep base_weight frozen and learn alpha & beta.
    W' = alpha * W + beta
    """
    def __init__(self, module):
        super().__init__()
        self.module = module
        W = module.weight.data.clone()
        self.register_buffer('base_weight', W)  # frozen base
        # alpha shape matches weight shape, init to ones
        self.alpha = nn.Parameter(torch.ones_like(W))
        self.beta = nn.Parameter(torch.zeros_like(W))
        # if bias exists, keep normal bias param
        if getattr(module, 'bias', None) is not None:
            self.bias = module.bias
        else:
            self.bias = None

    def forward(self, x):
        Wprime = self.alpha * self.base_weight + self.beta
        # use functional conv/linear depending on module type
        if isinstance(self.module, nn.Conv2d):
            return nn.functional.conv2d(x, Wprime, bias=self.bias,
                                        stride=self.module.stride, padding=self.module.padding)
        elif isinstance(self.module, nn.Linear):
            return nn.functional.linear(x, Wprime, bias=self.bias)
        else:
            raise NotImplementedError("Wrap conv/linear only.")
