import torch
import torch.nn as nn
import math

def compared_version(ver1, ver2):
    """
    :param ver1
    :param ver2
    :return: ver1< = >ver2 False/True
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")
    
    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1
    
    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True

class ChrPosEncoding(nn.Module):

    def __init__(self, d_model, fam_info):
        super(ChrPosEncoding, self).__init__()
        assert d_model % 2 == 0, 'd_model must be even'  # d_model must be even so that we can divide by 2, the first half for chromosome number and the second half for position

        marker_num = fam_info.shape[0]

        div_term = (torch.arange(0, d_model//2, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model//2)).exp()
        pe = torch.zeros(marker_num, d_model).float()
        pe_chr = torch.zeros(marker_num, d_model//2)
        pe_pos = torch.zeros(marker_num, d_model//2)
        pe_chr.require_grad = False

        chr_num = fam_info[:, 0]
        pos = fam_info[:, 1]

        pe_chr[:, 0::2] = torch.sin(chr_num * div_term)
        pe_chr[:, 1::2] = torch.cos(chr_num * div_term)

        pe_pos[:, 0::2] = torch.sin(pos * div_term)
        pe_pos[:, 1::2] = torch.cos(pos * div_term)

        pe[:, :d_model//2] = pe_chr
        pe[:, d_model//2:] = pe_pos

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, max_chr=20, dropout=0.0):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        chrom = torch.arange(0, max_chr).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        if max_chr > 1:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

        else:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.chromosome_embedding = PositionalEmbedding(d_model=d_model)
        #self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
        #                                            freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
        #    d_model=d_model, embed_type=embed_type, freq=freq)
        

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) #self.temporal_embedding(x_mark) + 
        return self.dropout(x)

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, fam_info=None):
        super(DataEmbedding_wo_pos, self).__init__()

        #self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = ChrPosEncoding(d_model, fam_info) #PositionalEmbedding(d_model=d_model)
        #self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
        #                                            freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
        #    d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
        self.localConv = locallyConnected1d(in_channels=1, out_channels=d_model, input_length=fam_info.shape[0], kernel_size=3, stride=1, bias=False)

    def forward(self, x, x_mark):

        x = self.position_embedding(x)
        x = self.dropout(x)
        x = self.localConv(x.unsqueeze(1)).squeeze(1)
        #x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark) 
        return x


class locallyConnected1d(nn.Module):
    def __init__(self, in_channels, out_channels, input_length, kernel_size, stride, bias=True):
        super(locallyConnected1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.input_length = input_length
        #Calculate padding size to ensure the divided result is an integer
        self.padding = (self.stride * (self.input_length - 1) - self.input_length + self.kernel_size) // 2
        self.numBlocks = self.input_length // self.stride + 1

        self.weight = nn.Parameter(torch.Tensor(self.numBlocks, self.in_channels, self.kernel_size, self.out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.numBlocks, self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        #Assume input shape is [batch_size, in_channels, input_length], output shape is [batch_size, out_channels, numBlocks]
        # Add padding to the input 
        x = nn.functional.pad(x, (self.padding, self.padding))
        x = x.unfold(2, self.kernel_size, self.stride).permute(0, 2, 1, 3).contiguous() # [batch_size, numBlocks, in_channels, kernel_size]
        # Multiple the weight by kernels, (B, numBlocks, in_channels, kernel_size) * (numBlocks, in_channels, kernel_size, out_channels) -> (B, numBlocks, out_channels)
        x = torch.einsum('bnil,nilo->bonl', x, self.weight)
        # Sum the output of each kernel and reduce the last dimension
        x = x.sum(dim=-1) # [batch_size, out_channels, numBlocks]

        if self.bias is not None:
            x += self.bias.unsqueeze(0)
        
        return x # [batch_size, out_channels, numBlocks]