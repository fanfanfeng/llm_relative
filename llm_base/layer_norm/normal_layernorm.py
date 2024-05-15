import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self,num_features,eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True,unbiased=False)
        normalized_x = (x - mean) / (std + self.eps)
        return self.gamma * normalized_x + self.beta

if __name__ == '__main__':
    batch_size = 2
    seqlen = 3
    hidden_dim = 4

    # 初始化一个随机tensor
    x = torch.randn(batch_size,seqlen,hidden_dim)
    print(x)

    # 初始化LayerNorm
    layer_norm  = LayerNorm(num_features=hidden_dim)
    output_tensor = layer_norm(x)
    print("output after layer norm:\n,",output_tensor)

    torch_layer_norm = torch.nn.LayerNorm(normalized_shape=hidden_dim)
    torch_output_tensor = torch_layer_norm(x)
    print("output after torch layer norm:\n",torch_output_tensor)
