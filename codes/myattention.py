import torch
from torch import nn
import copy,math
from torch.autograd import Variable
import torch.nn.functional as F
        

class myAttention(nn.Module):
    def __init__(self):
        super(myAttention, self).__init__()
        self.WQ = nn.Linear(16, 16)
        self.WK = nn.Linear(16, 16)
        self.sm = nn.Softmax(dim=1)

    def forward(self, input_mlp1, input_mlp2, input_mlp3, input_mlp4):
        input = torch.cat([input_mlp1.unsqueeze(2), input_mlp2.unsqueeze(2), input_mlp3.unsqueeze(2), input_mlp4.unsqueeze(2)], dim=2)
        M = torch.bmm(input.permute(0,2,1), input)
        att = self.sm(M)
        g = torch.bmm(input, att)
        output = g.sum(dim=2)
        return output,att


class myMHAttention(nn.Module):
    def __init__(self,N, M, d_model,d_ff):

        super(myMHAttention,self).__init__()
        self.self_attn = MultiHeadAttention(M, d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.encoder_layer = EncoderLayer(d_model, self.self_attn, self.ff)
        self.WQ = nn.Linear(16, 16)
        self.WK = nn.Linear(16, 16)
        self.sm = nn.Softmax(dim=1)
        self.encoder = Encoder(self.encoder_layer, N)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  #这是   #这首
    
    def forward(self, input_mlp1, input_mlp2, input_mlp3, input_mlp4):
        x = torch.cat(
            [input_mlp1.unsqueeze(2), input_mlp2.unsqueeze(2), input_mlp3.unsqueeze(2), input_mlp4.unsqueeze(2)], dim=2)
        batch = x.size(0)
        mask=None
        x=x.permute(0,2,1)
        x = self.encoder(x, mask)
        x=x.permute(0,2,1)
        temp = x
        print(temp.shape)
        temp = temp.permute(0, 2, 1)
        print(temp.shape)
        M = torch.bmm(x.permute(0, 2, 1), x)
        print(M.shape)
        att = self.sm(M)
        g = torch.bmm(x, att)
        output = g.sum(dim=2)
        return output,att
        
        
def clones(module, N):
    """Produce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SkipConnection(nn.Module):
    """residual connection"""

    def __init__(self, size):
        super().__init__()
        self.norm = nn.BatchNorm1d(size)

    def forward(self, x, sublayer):
        """"Apply residual connection to any sublayer with the same size. Add & Norm"""
        x = x + sublayer(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class EncoderLayer(nn.Module):
    """"Encoder is made up of self-attention and feed forward"""

    def __init__(self, size, self_attn, feed_forward):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SkipConnection(size), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # return self.sublayer[1](x, self.feed_forward)
        return x

def attention(query, key, value, mask=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # scores = scores.masked_fill(mask == 0, -1e9)
        scores = scores.masked_fill(mask == 0, -math.inf)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, M, d_model):
        """Take in model size and number of heads."""
        super().__init__()
        assert d_model % M == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // M
        self.M = M
       
        self.linears = clones(nn.Linear(d_model, d_model, bias=False), 4)  # encoder: 3 + 1
       
    def forward(self, query, key, value, mask=None):
        """"Implements Figure 2"""
        # Same mask applied to all M heads.  awesome!
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => M x d_k
        query, key, value = [l(x).view(nbatches, -1, self.M, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, p_attn = attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.M * self.d_k)
        return self.linears[-1](x)


class FeedForward(nn.Module):
    """"Implements FFN equation."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))


class Encoder(nn.Module):
    """Core encoder is a stack of N layers."""

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return x

