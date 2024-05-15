import torch
from torch import Size
from typing import Union,List
from torch.nn import LayerNorm

Number_Layer = 1000 # Encoder/Decoder

class DeepNorm(torch.nn.Module):
    def __init__(self,normalized_shape: Union[int,List[int],Size],eps: float = 1e-5, elementwise_affine: bool = True):
        '''
          Deep Layer Normalization
        :param normalized_shape: input shape from an expected input of size
        :param eps: a value added to the denominator for numerical stability
        :param elementwise_affine:  a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
        '''
        super(DeepNorm,self).__init__()

        self.alpha = (2 * Number_Layer) ** 0.25
        self.layernorm = LayerNorm(normalized_shape,eps=eps)

    def forward(self,x):
        x_normed = self.layernorm(x)
        return self.alpha * x + x_normed