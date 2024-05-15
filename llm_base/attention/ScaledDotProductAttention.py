from torch import nn
import torch
import numpy as np

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled Dot-Product Attention
    '''

    def __init__(self,scale):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self,q,k,v,mask=None):
        # 1 Matmul
        u = torch.bmm(q,k.transpose(1,2))

        # 2 Scale
        u = u / self.scale

        # 3 Mask
        if mask is not None:
            u = u.masked_fill(mask,-np.inf)

        # 4 Softmax
        attn = self.softmax(u)

        # 5 Output
        output = torch.bmm(attn,v)

        return attn,output

if __name__ == '__main__':
    n_q,n_k,n_v = 2,4,4
    d_q,d_k,d_v =128,128,64

    batch = 100
    q = torch.randn(batch,n_q,d_q)
    k = torch.randn(batch,n_k,d_k)
    v = torch.randn(batch,n_v,d_v)
    mask = None#torch.zeros(batch,n_q,n_k).bool()

    attention = ScaledDotProductAttention(scale=np.power(d_k,0.5))
    attn,output = attention(q,k,v,mask=mask)

    print(attn)
    print(output)