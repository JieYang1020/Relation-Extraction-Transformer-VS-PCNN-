# coding: utf-8
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from .BasicModule import BasicModule

####################PositionalEncoding层####################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.0)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0., max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             (math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        standard_value = self.pe[:, :x.size(1)].expand(x.size(0), 82, 64)
        x = x + torch.autograd.Variable(standard_value,
                                        requires_grad=False)
        return self.dropout(x)
####################mask####################
#encoding的时候需要先padding
def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)
    return pad_mask
####################scaled dot_product attention####################
class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism"""
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim = 2)
    
    def forward(self, q, k, v, scale = None, attn_mask = None):
        '''前向传播。scale:缩放因子（1/(根号K)），一个浮点标量。返回：上下文张量和
        attention张量'''
        attention = torch.matmul(q, k.transpose(1, 2))
        #attention.shape[batch_size*num_heads, max_len, max_len]
        #attn_mask.shape[batch_size*num_heads, max_len, max_len]
        if scale:
            attention = attention * scale
        # if attn_mask:
            #给需要mask的地方设置负无穷
        attention = attention.masked_fill_(attn_mask, -np.inf)
        #计算softmax
        attention = self.softmax(attention)
        #添加dropout
        attention = self.dropout(attention)
        #和V做点积
        context = torch.bmm(attention, v)
        return context, attention
####################残差连接####################
def residual(sublayer_fn, x):
    return sublayer_fn(x)+x
####################multi-head attention####################
#将Q,K,V拆分为2份(heads=2)，每份分别进行scaled dot-product attention
class MultiHeadAttention(BasicModule):
    def __init__(self, opt):
        super(MultiHeadAttention, self).__init__()
        #dim_per_head:q, k, v向量的长度.Wq, Wk, Wv 的矩阵尺寸为model_dim/dim_per_head
        self.opt = opt
        self.dropout = 0.0
        self.linear_k = nn.Linear(self.opt.model_dim, self.opt.model_dim)
        self.linear_q = nn.Linear(self.opt.model_dim, self.opt.model_dim)
        self.linear_v = nn.Linear(self.opt.model_dim, self.opt.model_dim)
        self.dot_product_attention = ScaledDotProductAttention(self.dropout)
        self.linear_final = nn.Linear(self.opt.model_dim, self.opt.model_dim)
        self.dropout = nn.Dropout(self.dropout)
        # multi_head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(self.opt.model_dim)
    
    def forward(self, key, value, query, attn_mask=None):
        #残差连接，增加任意常数，求导为1，避免反向传播时的梯度消失
        residual = query
        self.dim_per_head = self.opt.model_dim // self.opt.num_heads
        batch_size = key.size(0)
        #linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        #split by heads
        key = key.view(batch_size*self.opt.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size*self.opt.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size*self.opt.num_heads, -1, self.dim_per_head)
        # if attn_mask:
        #repeat是为了能在dot_product_attention中做mask_fill
        attn_mask = attn_mask.repeat(self.opt.num_heads, 1, 1)
        scale = (key.size(-1) // self.opt.num_heads) ** -0.5
        #调用之前的scaled dot product attention
        context, attention = self.dot_product_attention(query, key, value,
                                                        scale, attn_mask)
        #将2个头的结果连接
        context = context.view(batch_size, -1, self.opt.model_dim)
        #final linear projection
        output = self.linear_final(context)
        output = self.dropout(output)
        #add residual and norm layer
        residual = residual.squeeze(1)
        output = self.layer_norm(residual + output)
        return output, attention
####################position-wise feed-forwad network####################
class PositionalWiseFeedForward(BasicModule):
    def __init__(self, opt):
        super(PositionalWiseFeedForward, self).__init__()
        self.opt = opt
        self.dropout = 0.0
        self.w1 = nn.Conv1d(self.opt.model_dim, self.opt.ffn_dim, 1)
        self.w2 = nn.Conv1d(self.opt.ffn_dim, self.opt.model_dim, 1)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.opt.model_dim)
    def forward(self, x):
        # torch.Size([batch_size, max_len, model_dim])
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1,2))
        #add residual and norm layer
        output = self.layer_norm(x + output)
        return output
####################EncoderLayer层(一个encoder框架可以有多个encoderlayer层，
#用num_layers选择)####################
class EncoderLayer(BasicModule):
    def __init__(self, opt):
        super(EncoderLayer, self).__init__()
        self.opt = opt
        self.dropout = 0.0
        self.attention = MultiHeadAttention(self.opt)
        self.feed_forward = PositionalWiseFeedForward(self.opt)
    def forward(self, embedding_output,self_attention_mask, attn_mask = None):
        #embedding_output.shape=[batch_size,max_len,model_dim]
        #padding_mask.shape=[batch_size,max_len,max_len]
        #mask应该是只对词本身padding
        context, attention = self.attention(embedding_output, embedding_output,
                                            embedding_output, self_attention_mask)
        #feed forward network
        output = self.feed_forward(context)
        return output, attention
class Encoder(BasicModule):
    '''多层EncoderLayer组成Encoder'''
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.opt) for
                                            _ in range(self.opt.num_layers)])
        self.seq_embedding = nn.Embedding(self.opt.vocab_size + 1, self.opt.model_dim,
                                          padding_idx=0)
        self.pos_embedding = PositionalEncoding(self.opt.model_dim, self.opt.max_len)
    def forward(self, inputs, insPF1, insPF2):
        output_context = self.seq_embedding(inputs)
        # 应先做了positional embedding(为了维度扩展), 再做positional encoding, 再将PF1,PF2合并
        output_pos1 = self.pos_embedding(self.seq_embedding(insPF1))
        output_pos2 = self.pos_embedding(self.seq_embedding(insPF2))
        #三个结果衔接,衔接后size为[batch_size, max_len, model_dim]
        embedding_output = output_context + (output_pos1 + output_pos2)
        #q,k,v的size应为[batch_size, max_len, h, model_dim]
        embedding_output = embedding_output.unsqueeze(1)
        self_attention_mask = padding_mask(inputs, inputs)
        attentions = []
        for encoder in self.encoder_layers:
            #分别进行mask,多头，ffn等层
            output, attention = encoder(embedding_output, self_attention_mask)
            attentions.append(attention)
        return output, attentions
###################模型入口，先衔接Encoder类####################
class Transformer_self(BasicModule):
    def __init__(self, opt):
        super(Transformer_self, self).__init__()
        self.model_name = 'Transformer_self'
        self.opt = opt
        self.encoder = Encoder(self.opt)
        self.linear = nn.Linear(self.opt.max_len, self.opt.rel_num, bias=False)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, inputs):
        insEnt, _, insX, insPFs, insPool, insMasks = inputs
        #insX是input_vector,size = [batch_size,max_len]
        #insPFs是位置信息, size = [batch_size, 2, max_len]
        #拆分后insPF1,insPF2的size都为[batch_size, max_len]
        insPF1, insPF2 = [i.squeeze(1) for i in torch.split(insPFs, 1, 1)]
        #进入encoder框架
        output, enc_self_attn = self.encoder(insX, insPF1, insPF2)
        #全连接前降维(使用max将max_len去掉),类似kernel_size = 82的一维池化
        output = torch.max(output, dim=2).values
        # output = output.view(output.size(0), -1)
        output = self.linear(output)
        output = self.softmax(output)
        return output

